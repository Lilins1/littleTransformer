from littleTransformer import DecoderOnlyTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import tqdm
import json
import os
from datetime import datetime, timedelta
import sys
import wandb
from huggingface_hub import HfApi, login
import time

torch.autograd.set_detect_anomaly(True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer.Tokenizer import BPETokenizer

# ------------------------------
# 折中方案：部分预计算估算总token数
# ------------------------------
def estimate_total_tokens(file_path, tokenizer, sample_size=1000, max_length=100_000_000):
    """只计算前N个样本的token数，估算总量（大幅减少预计算时间）"""
    total_samples = 0  # 总样本数
    sample_tokens = 0  # 抽样样本的总token数
    current_length = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = list(data.items())  # 转为列表方便抽样
    
    # 1. 先统计总样本数和抽样计算平均token数
    for i, (key, value) in enumerate(items):
        record_text = " ".join([
            f"QUESTION: {value['QUESTION']}",
            f"CONTEXT: {' '.join(value['CONTEXTS'])}",
            f"LONG_ANSWER: {value['LONG_ANSWER']}",
            "<EOS> <SOS>"
        ])
        
        # 控制文本总长度上限
        if current_length + len(record_text) > max_length:
            break
        current_length += len(record_text)
        total_samples += 1
        
        # 只对前sample_size个样本计算token数（估算平均）
        if i < sample_size:
            tokens = tokenizer.encode(record_text, max_len=100000)
            sample_tokens += len(tokens)
    
    # 2. 估算总token数 = 总样本数 × 平均每个样本的token数
    if total_samples == 0:
        return 0
    avg_tokens_per_sample = sample_tokens / min(sample_size, total_samples)  # 避免除以0
    estimated_total = int(total_samples * avg_tokens_per_sample)
    print(f"基于{min(sample_size, total_samples)}个样本估算：总样本数={total_samples}，平均token数={avg_tokens_per_sample:.1f}")
    return estimated_total

# ------------------------------
# 1. 数据生成器和数据集（支持动态停止）
# ------------------------------
def text_generator(file_path, max_length=100_000_000, stop_signal=None):
    """支持动态停止的文本生成器（用于控制每个epoch的数据量）"""
    current_length = 0
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for key, value in data.items():
        # 检查是否需要提前停止（用于动态调整）
        if stop_signal and stop_signal['should_stop']:
            stop_signal['should_stop'] = False  # 重置信号
            break
            
        record_text = " ".join([
            f"QUESTION: {value['QUESTION']}",
            f"CONTEXT: {' '.join(value['CONTEXTS'])}",
            f"LONG_ANSWER: {value['LONG_ANSWER']}",
            "<EOS> <SOS>"
        ])
        
        if current_length + len(record_text) > max_length:
            break
        current_length += len(record_text)
        yield record_text

def token_generator(text_gen, tokenizer, buffer_size=10000, max_token_len=100000):
    token_buffer = []
    for text in text_gen:
        tokens = tokenizer.encode(text, max_len=max_token_len)
        token_buffer.extend(tokens)
        
        while len(token_buffer) >= buffer_size:
            yield token_buffer[:buffer_size]
            token_buffer = token_buffer[buffer_size:]
    
    if token_buffer:
        yield token_buffer

class StreamingTextDataset(IterableDataset):
    def __init__(self, token_gen, seq_len=100):
        self.token_gen = token_gen
        self.seq_len = seq_len
        self.token_buffer = []

    def __iter__(self):
        for token_batch in self.token_gen:
            self.token_buffer.extend(token_batch)
            
            while len(self.token_buffer) >= self.seq_len:
                yield torch.tensor(self.token_buffer[:self.seq_len], dtype=torch.long)
                self.token_buffer = self.token_buffer[1:]

# ------------------------------
# 2. 训练器（支持动态调整批次）
# ------------------------------
class Trainer:
    def __init__(self, model, optimizer, criterion, device, config, wandb_run=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.save_path = config['save_path']
        self.best_val_loss = float('inf')
        self.wandb_run = wandb_run
        self.last_save_time = time.time()
        self.actual_train_tokens = 0  # 记录实际处理的token数（用于动态调整）
        
        self.hf_api = HfApi() if config['upload_to_hub'] else None
        if config['upload_to_hub'] and config['hub_token']:
            login(config['hub_token'], add_to_git_credential=True)

    def _upload_to_hub(self, file_path):
        try:
            self.hf_api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),
                repo_id=self.config['hub_model_id'],
                repo_type="model"
            )
            print(f"✅ 已上传 {file_path} 到 Hugging Face Hub")
        except Exception as e:
            print(f"❌ Hugging Face 上传失败: {str(e)}")

    def _log_and_upload(self, epoch, train_loss, val_loss):
        if self.wandb_run:
            self.wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": self.best_val_loss,
                "actual_train_tokens": self.actual_train_tokens  # 记录实际处理的token数
            })
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "actual_train_tokens": self.actual_train_tokens
        }
        with open("training_metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        if self.config['upload_to_hub']:
            self._upload_to_hub("training_metrics.jsonl")

    def _save_model(self, epoch, is_best=False):
        os.makedirs("Models", exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        if not is_best:
            # 创建每个 epoch 的子目录
            epoch_dir = f"Models/model_epoch_{epoch}_{timestamp}"
            os.makedirs(epoch_dir, exist_ok=True)
            save_name = f"{epoch_dir}/model_epoch_{epoch}_{timestamp}.pt"
        else:
            save_name = self.save_path  # 最佳模型保存到固定路径

        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if is_best:
            save_dict['loss'] = self.best_val_loss

        torch.save(save_dict, save_name)

        if self.config.get('upload_to_hub', False):
            try:
                self._upload_to_hub(save_name)
            except Exception as e:
                print(f"上传到 Hub 失败: {e}")

        return save_name

    def _train_epoch(self, train_loader, max_batches):
        self.model.train()
        self.actual_train_tokens = 0  # 重置计数
        epoch_loss, batch_count = 0, 0
        
        with tqdm.tqdm(total=max_batches, desc=f"训练中") as pbar:
            for batch in train_loader:
                if batch_count >= max_batches:
                    break
                
                # 累加实际处理的token数（用于后续调整）
                self.actual_train_tokens += batch.numel()
                
                src = batch.to(self.device)
                inputs, targets = src[:, :-1], src[:, 1:]
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    targets.reshape(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                pbar.update(1)
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        
        return epoch_loss / batch_count if batch_count > 0 else 0

    def _evaluate(self, val_loader, max_batches):
        self.model.eval()
        val_loss, batch_count = 0, 0
        
        with tqdm.tqdm(total=max_batches, desc=f"验证中") as pbar:
            with torch.no_grad():
                for batch in val_loader:
                    if batch_count >= max_batches:
                        break
                    
                    src = batch.to(self.device)
                    inputs, targets = src[:, :-1], src[:, 1:]
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(
                        outputs.reshape(-1, outputs.shape[-1]),
                        targets.reshape(-1)
                    )
                    val_loss += loss.item()
                    batch_count += 1
                    pbar.update(1)
                    pbar.set_postfix({"val_batch_loss": f"{loss.item():.4f}"})
        
        return val_loss / batch_count if batch_count > 0 else 0

    def train(self, num_epochs, train_loader, val_loader, train_max_batches, val_max_batches):
        for epoch in range(num_epochs):
            print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
            
            # 动态使用当前的批次上限
            train_loss = self._train_epoch(train_loader, train_max_batches)
            val_loss = self._evaluate(val_loader, val_max_batches)
            
            self._log_and_upload(epoch+1, train_loss, val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                saved_path = self._save_model(epoch+1, is_best=True)
                print(f"📌 最佳模型已保存至 {saved_path} (验证损失: {val_loss:.4f})")
            
            # 定时保存
            current_time = time.time()
            if (epoch + 1) % self.config['save_interval_epochs'] == 0 or \
               current_time - self.last_save_time >= self.config['save_interval_seconds']:
                
                saved_path = self._save_model(epoch+1)
                print(f"⏱️ 定时保存模型至 {saved_path} (Epoch {epoch+1})")
                self.last_save_time = current_time

# ------------------------------
# 主训练流程（折中方案核心）
# ------------------------------
def train_process(config):
    # 初始化W&B
    wandb_run = wandb.init(
        project=config['wandb_project'],
        name=config.get('wandb_run_name'),
        config=config
    ) if config['use_wandb'] else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if wandb_run:
        wandb_run.config.update({"device": str(device)})

    # 初始化分词器
    tokenizer = BPETokenizer(config['vocab_path'])
    config['vocab_size'] = tokenizer.vocab_size
    config['pad_idx'] = tokenizer.PAD_ID
    print(f"分词器初始化完成。词汇表大小: {config['vocab_size']}")

    # ------------------------------
    # 折中方案：快速估算批次（核心）
    # ------------------------------
    # 1. 快速估算训练集总token数（只抽样前N个样本）
    print("快速估算训练集token数（折中方案）...")
    estimated_train_tokens = estimate_total_tokens(
        file_path=config['data_path'],
        tokenizer=tokenizer,
        sample_size=config['estimate_sample_size'],  # 可调整抽样量（默认1000）
        max_length=config['context_length']
    )
    # 计算初始批次上限（比估算值略大，避免提前结束）
    initial_train_batches = max(1, estimated_train_tokens // config['max_len'] // config['batch_size'] * 1.2)
    initial_train_batches = int(initial_train_batches)
    print(f"估算训练集token数: {estimated_train_tokens:,}，初始批次上限: {initial_train_batches}")

    # 2. 同理估算验证集
    print("快速估算验证集token数...")
    estimated_val_tokens = estimate_total_tokens(
        file_path=config['data_path'],
        tokenizer=tokenizer,
        sample_size=config['estimate_sample_size'] // 5,  # 验证集抽样更少
        max_length=config['context_length'] // 10
    )
    initial_val_batches = max(1, estimated_val_tokens // config['max_len'] // config['batch_size'] * 1.2)
    initial_val_batches = int(initial_val_batches)
    print(f"估算验证集token数: {estimated_val_tokens:,}，初始批次上限: {initial_val_batches}")

    # 3. 流式数据加载（支持动态停止）
    stop_signal = {'should_stop': False}  # 用于控制生成器停止的信号
    
    train_text_gen = text_generator(
        file_path=config['data_path'],
        max_length=config['context_length'],
        stop_signal=stop_signal
    )
    train_token_gen = token_generator(
        train_text_gen, 
        tokenizer, 
        buffer_size=50000,
        max_token_len=config['max_len'] * 10
    )

    val_text_gen = text_generator(
        file_path=config['data_path'],
        max_length=config['context_length'] // 10,
        stop_signal=stop_signal
    )
    val_token_gen = token_generator(
        val_text_gen, 
        tokenizer, 
        buffer_size=10000,
        max_token_len=config['max_len'] * 10
    )

    # 数据集和加载器
    train_dataset = StreamingTextDataset(train_token_gen, seq_len=config['max_len'])
    val_dataset = StreamingTextDataset(val_token_gen, seq_len=config['max_len'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=0)
    print("流式DataLoader已准备好，开始训练...")

    # 模型初始化
    model = DecoderOnlyTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    ).to(device)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 训练初始化
    criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        wandb_run=wandb_run
    )

    # 启动训练（使用初始批次上限）
    trainer.train(
        num_epochs=config['num_epochs'],
        train_loader=train_loader,
        val_loader=val_loader,
        train_max_batches=initial_train_batches,
        val_max_batches=initial_val_batches
    )
    print("\n--- 训练完成 ---")

    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    config = {
        # 基础配置
        'vocab_path': '../tokenizer/vocab_test_set.npy',
        'data_path': r'C:\Users\Ruizhe\Desktop\Study\ID2221\Project\Data\pubmedqa\ori_pqau.json',
        'save_path': 'best_model.pt',
        'context_length': 1000_000_000,
        'max_len': 100,
        
        # 折中方案参数（核心）
        'estimate_sample_size': 1000,  # 抽样多少个样本估算token数（可调）
        
        # 模型参数
        'd_model': 768,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 3072,
        'dropout': 0.1,
        
        # 训练参数
        'batch_size': 32,
        'learning_rate': 5e-5,
        'num_epochs': 2,
        
        # 保存配置
        'save_interval_epochs': 1,
        'save_interval_seconds': 21600,
        
        # W&B和HF配置
        'use_wandb': True,
        'wandb_project': 'pubmedqa-training',
        'wandb_run_name': f"balanced-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        'upload_to_hub': False,
        'hub_model_id': 'your_username/pubmedqa-model',
        'hub_token': 'your_hf_token'
    }
    
    train_process(config)
