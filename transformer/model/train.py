from littleTransformer import PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward, EncoderLayer, DecoderLayer, DecoderOnlyTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import numpy as np
import json
import os
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer.Tokenizer import BPETokenizer

# 导入W&B库
import wandb

class Trainer:
    def __init__(self, model, optimizer, criterion, device, save_path="best_model.pt", 
                 wandb_run=None):  # 新增W&B运行实例参数
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.wandb_run = wandb_run  # 保存W&B运行实例

    def _log_metrics(self, epoch, train_loss, val_loss):
        """使用W&B记录指标"""
        if self.wandb_run:
            # 直接通过W&B记录指标，自动同步到云端
            self.wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,  # 修正原代码中的笔误（train_loss_loss）
                "val_loss": val_loss,
                "best_val_loss": self.best_val_loss
            })
            print(f"已将第{epoch}轮指标上传到W&B")

    def _train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        for batch in tqdm.tqdm(train_loader, desc="Training"):
            src = batch[0].to(self.device)
            inputs = src[:, :-1]
            targets = src[:, 1:]
            
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
        return epoch_loss / len(train_loader)

    def _evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc="Evaluating"):
                src = batch[0].to(self.device)
                inputs = src[:, :-1]
                targets = src[:, 1:]
                
                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    targets.reshape(-1)
                )
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def train(self, num_epochs, train_loader, val_loader):
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                }, self.save_path)
                print(f"  模型已保存（验证损失降低至 {val_loss:.4f}）")
            
            # 记录指标到W&B
            self._log_metrics(epoch+1, train_loss, val_loss)


def load_and_concatenate_text(file_path, max_length=100_000):
    text_list = []
    current_length = 0
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, value in data.items():
        q = f"QUESTION: {value['QUESTION']}"
        c = f"CONTEXT: {' '.join(value['CONTEXTS'])}"
        la = f"LONG_ANSWER: {value['LONG_ANSWER']}"
        record_text = " ".join([q, c, la])
        
        if current_length + len(record_text) > max_length:
            break
        text_list.append(record_text)
        current_length += len(record_text)
    text = " <EOS> <SOS> ".join(text_list)
    print(f"加载并拼接完成，文本长度：{len(text)}字符")
    return text


def train_process(config):
    # 初始化W&B（核心改动）
    wandb_run = wandb.init(
        project=config['wandb_project'],  # W&B项目名称
        name=config.get('wandb_run_name'),  # 运行名称（可选）
        config=config  # 自动记录超参数
    ) if config['use_wandb'] else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wandb_run:
        wandb_run.config.update({"device": str(device)})  # 记录使用的设备

    # 初始化分词器
    tokenizer = BPETokenizer(config['vocab_path'])
    config['vocab_size'] = tokenizer.vocab_size
    config['pad_idx'] = tokenizer.PAD_ID
    print(f"分词器初始化完成。词汇表大小: {config['vocab_size']}")

    # 加载并编码数据
    raw_text_data = load_and_concatenate_text(file_path=config['data_path'], max_length=config['context_lengh'])
    print("开始对全部文本进行编码...")
    all_token_ids = tokenizer.encode_corpus(raw_text_data) 
    print(f"文本编码完成。总 Token 数量: {len(all_token_ids)}")

    # --- STEP 3 & 4: 创建训练/验证样本 (修正版) ---
    
    # 1. 先分割 token ID 序列，避免数据泄露
    num_tokens = len(all_token_ids)
    split_idx = int(num_tokens * 0.9)
    
    train_token_ids = all_token_ids[:split_idx]
    val_token_ids = all_token_ids[split_idx:]
    print(f"Token 序列已分割: {len(train_token_ids)}个用于训练, {len(val_token_ids)}个用于验证")

    # 2. 分别为训练集和验证集创建样本
    seq_len = config['max_len']
    
    def create_chunks(token_ids, chunk_size):
        chunks = []
        for i in range(0, len(token_ids) - chunk_size, 1):
            chunk = token_ids[i : i + chunk_size]
            chunks.append(torch.tensor(chunk, dtype=torch.long))
        if not chunks: # 如果token数量太少，至少创建一个样本
             return torch.empty(0, chunk_size, dtype=torch.long)
        return torch.stack(chunks)

    print("正在为训练集创建数据块...")
    train_dataset_tensor = create_chunks(train_token_ids, seq_len)
    print("正在为验证集创建数据块...")
    val_dataset_tensor = create_chunks(val_token_ids, seq_len)

    print(f"已创建 {train_dataset_tensor.shape[0]} 个训练样本和 {val_dataset_tensor.shape[0]} 个验证样本")

    # 3. 创建 DataLoader
    train_data = TensorDataset(train_dataset_tensor)
    val_data = TensorDataset(val_dataset_tensor)
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])
    print("真实数据 DataLoader 已准备好")

    # 准备模型
    model = DecoderOnlyTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    ).to(device)
    print(f"模型已创建并移动到 {device}")
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {param_count:,}")
    if wandb_run:
        wandb_run.config.update({"param_count": param_count})  # 记录参数量

    # 初始化训练器（传入W&B运行实例）
    criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_path=config['save_path'],
        wandb_run=wandb_run  # 传递W&B实例
    )

    # 启动训练
    print("\n--- 开始使用真实数据进行训练 ---")
    trainer.train(num_epochs=config['num_epochs'], train_loader=train_loader, val_loader=val_loader)
    print("--- 训练完成 ---")

    # 结束W&B运行
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    config = {
        'vocab_path': '../tokenizer/vocab_test_set.npy',
        'data_path': r'C:\Users\Ruizhe\Desktop\Study\ID2221\Project\Data\pubmedqa\ori_pqau.json',
        'save_path': 'best_model.pt',
        # W&B配置（新增）
        'use_wandb': True,  # 是否启用W&B
        'wandb_project': 'pubmedqa-training',  # 你的W&B项目名称
        'wandb_run_name': f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",  # 自动生成运行名称
        # 原有超参数
        'vocab_size': None,
        'pad_idx': None,
        'context_lengh': 1000_000_000, # 1000_000_000
        'd_model': 768,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 3072,
        'max_len': 100,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 5
    }
    
    train_process(config)