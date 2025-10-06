import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import GradScaler, autocast

import math
import tqdm
import json
import os
import sys
import time
from datetime import datetime
import wandb
from huggingface_hub import HfApi, login

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer.Tokenizer import BPETokenizer
from littleTransformer import DecoderOnlyTransformer

# ------------------------------
# 数据估算与生成
# ------------------------------
def estimate_total_tokens(file_path, tokenizer, sample_size=1000, max_length=1_000_000_000):
    """通过抽样估算总 token 数。"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = list(data.items())
    
    total_samples = 0
    sample_tokens = 0
    current_length = 0
    
    for i, (key, value) in enumerate(items):
        record_text = f"QUESTION: {value['QUESTION']} CONTEXT: {' '.join(value['CONTEXTS'])} LONG_ANSWER: {value['LONG_ANSWER']} <EOS> <SOS>"
        if current_length + len(record_text) > max_length:
            break
        current_length += len(record_text)
        total_samples += 1
        if i < sample_size:
            tokens = tokenizer.encode(record_text)
            sample_tokens += len(tokens)
    
    if total_samples == 0:
        return 0
    avg_tokens_per_sample = sample_tokens / min(sample_size, total_samples)
    estimated_total = int(total_samples * avg_tokens_per_sample)
    print(f"估算: 总样本={total_samples}, 平均tokens={avg_tokens_per_sample:.1f}, 总tokens≈{estimated_total}")
    return estimated_total

def text_generator(file_path, max_length=1_000_000_000):
    """生成文本记录。"""
    current_length = 0
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, value in data.items():
        record_text = f"QUESTION: {value['QUESTION']} CONTEXT: {' '.join(value['CONTEXTS'])} LONG_ANSWER: {value['LONG_ANSWER']} <EOS> <SOS>"
        if current_length + len(record_text) > max_length:
            break
        current_length += len(record_text)
        yield record_text

def token_generator(text_gen, tokenizer, buffer_size=10000):
    """将文本转换为 token 流。"""
    token_buffer = []
    for text in text_gen:
        tokens = tokenizer.encode(text)
        token_buffer.extend(tokens)
        while len(token_buffer) >= buffer_size:
            yield token_buffer[:buffer_size]
            token_buffer = token_buffer[buffer_size:]
    if token_buffer:
        yield token_buffer

class StreamingTextDataset(IterableDataset):
    """流式 token 数据集。"""
    def __init__(self, token_gen, seq_len=100):
        self.token_gen = token_gen
        self.seq_len = seq_len
        self.token_buffer = []

    def __iter__(self):
        for token_batch in self.token_gen:
            self.token_buffer.extend(token_batch)
            while len(self.token_buffer) >= self.seq_len:
                yield torch.tensor(self.token_buffer[:self.seq_len], dtype=torch.long)
                self.token_buffer = self.token_buffer[self.seq_len:]

# ------------------------------
# 训练器类
# ------------------------------
class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, config, wandb_run=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.best_val_loss = float('inf')
        self.wandb_run = wandb_run
        self.global_step = 0
        self.scaler = GradScaler()
        
        self.hf_api = HfApi() if config.get('upload_to_hub') else None
        if config.get('upload_to_hub') and config.get('hub_token'):
            login(config['hub_token'], add_to_git_credential=True)

    def _upload_to_hub(self, file_path):
        try:
            self.hf_api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),
                repo_id=self.config['hub_model_id'],
                repo_type="model"
            )
            print(f"✅ Uploaded {file_path} to Hugging Face Hub")
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")

    def _log_metrics(self, epoch, train_loss, val_loss):
        lr = self.optimizer.param_groups[0]['lr']
        if self.wandb_run:
            self.wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": self.best_val_loss,
                "learning_rate": lr
            }, step=self.global_step)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "learning_rate": lr
        }
        with open("training_metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        if self.config.get('upload_to_hub'):
            self._upload_to_hub("training_metrics.jsonl")

    def _save_model(self, epoch, is_best=False):
        os.makedirs("Models", exist_ok=True)
        filename = "best_model.pt" if is_best else f"model_epoch_{epoch}.pt"
        save_path = os.path.join("Models", filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.best_val_loss if is_best else None,
        }, save_path)
        
        if self.config.get('upload_to_hub'):
            self._upload_to_hub(save_path)
        return save_path

    def _train_epoch(self, train_loader, max_batches):
        self.model.train()
        total_loss = 0
        batch_count = 0
        interval_loss = 0
        log_interval = self.config.get('wandb_log_interval', 50)
        acc_steps = self.config.get('gradient_accumulation_steps', 1)
        grad_clip = self.config.get('grad_clip_norm', 1.0)

        with tqdm.tqdm(total=max_batches, desc="Training") as pbar:
            for i, batch in enumerate(train_loader):
                if i >= max_batches:
                    break
                
                src = batch.to(self.device)
                inputs, targets = src[:, :-1], src[:, 1:]
                if targets.shape[1] == 0:
                    pbar.update(1)
                    continue

                with autocast(device_type=self.device.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Unstable loss ({loss.item()}) in batch {i}, skipping.")
                    self.optimizer.zero_grad()
                    pbar.update(1)
                    continue

                loss = loss / acc_steps
                self.scaler.scale(loss).backward()

                if (i + 1) % acc_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    batch_loss = loss.item() * acc_steps
                    total_loss += batch_loss
                    interval_loss += batch_loss
                    batch_count += 1
                    self.global_step += 1
                    lr = self.optimizer.param_groups[0]['lr']

                    if self.wandb_run and self.global_step % log_interval == 0:
                        self.wandb_run.log({
                            "train_loss_batch": interval_loss / log_interval,
                            "learning_rate": lr,
                        }, step=self.global_step)
                        interval_loss = 0

                    pbar.set_postfix({"loss": f"{batch_loss:.4f}", "lr": f"{lr:.2e}"})

                pbar.update(1)

        return total_loss / batch_count if batch_count > 0 else 0

    def _evaluate(self, val_loader, max_batches):
        self.model.eval()
        total_loss = 0
        batch_count = 0

        with tqdm.tqdm(total=max_batches, desc="Validation") as pbar, torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max_batches:
                    break
                
                src = batch.to(self.device)
                inputs, targets = src[:, :-1], src[:, 1:]
                if targets.shape[1] == 0:
                    pbar.update(1)
                    continue

                with autocast(device_type=self.device.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    batch_count += 1
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                pbar.update(1)

        return total_loss / batch_count if batch_count > 0 else 0

    def train(self, num_epochs, train_loader, val_loader, train_max_batches, val_max_batches):
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            train_loss = self._train_epoch(train_loader, train_max_batches)
            val_loss = self._evaluate(val_loader, val_max_batches)
            
            self._log_metrics(epoch, train_loss, val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                path = self._save_model(epoch, is_best=True)
                print(f"Best model saved to {path} (val loss: {val_loss:.4f})")
            
            if epoch % self.config['save_interval_epochs'] == 0:
                path = self._save_model(epoch)
                print(f"Checkpoint saved to {path}")

# ------------------------------
# 主流程
# ------------------------------
def main(config):
    wandb_run = None
    if config['use_wandb']:
        wandb_run = wandb.init(
            project=config['wandb_project'],
            entity=config.get('wandb_entity'),
            name=config.get('wandb_run_name'),
            config=config
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_bf16_supported():
        print("bfloat16 supported.")

    tokenizer = BPETokenizer(config['vocab_path'])
    config['vocab_size'] = tokenizer.vocab_size
    config['pad_idx'] = tokenizer.PAD_ID
    print(f"Tokenizer: vocab_size={config['vocab_size']}, pad_id={config['pad_idx']}")

    est_train_tokens = estimate_total_tokens(config['data_path'], tokenizer, config['estimate_sample_size'], config['context_length'])
    train_batches = max(1, est_train_tokens // config['max_len'] // config['batch_size'])

    est_val_tokens = estimate_total_tokens(config['data_path'], tokenizer, config['estimate_sample_size'] // 5, config['context_length'] // 10)
    val_batches = max(1, est_val_tokens // config['max_len'] // config['batch_size'])
    print(f"Batches: train={train_batches}, val={val_batches}")

    train_text_gen = text_generator(config['data_path'], config['context_length'])
    train_token_gen = token_generator(train_text_gen, tokenizer, buffer_size=50000)
    train_dataset = StreamingTextDataset(train_token_gen, config['max_len'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])

    val_text_gen = text_generator(config['data_path'], config['context_length'] // 10)
    val_token_gen = token_generator(val_text_gen, tokenizer, buffer_size=10000)
    val_dataset = StreamingTextDataset(val_token_gen, config['max_len'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    print("DataLoaders ready.")

    model = DecoderOnlyTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)

    acc_steps = config.get('gradient_accumulation_steps', 1)
    total_steps = (train_batches // acc_steps) * config['num_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    print(f"Scheduler: total_steps={total_steps}, warmup={warmup_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    trainer = Trainer(model, optimizer, scheduler, criterion, device, config, wandb_run)
    trainer.train(config['num_epochs'], train_loader, val_loader, train_batches, val_batches)
    print("Training complete.")

    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    config = {
        'vocab_path': '../tokenizer/vocab_test_set.npy',
        'data_path': r'/home/ik2200-2025-g2/WorkZone/Ruizhe/Medical/pubmedqa/data/ori_pqau.json',
        'context_length': 1_000_000_000,
        'max_len': 100,
        'estimate_sample_size': 1000,
        'd_model': 768,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 3072,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 8,
        'warmup_ratio': 0.1,
        'gradient_accumulation_steps': 4,
        'grad_clip_norm': 1.0,
        'save_interval_epochs': 1,
        'use_wandb': True,
        'wandb_project': 'pubmedqa-training',
        'wandb_entity': 'liruizhe987-kth-royal-institute-of-technology',
        'wandb_run_name': f"final-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        'wandb_log_interval': 50,
        'upload_to_hub': False,
        'hub_model_id': 'your_username/pubmedqa-model',
        'hub_token': 'your_hf_token'
    }
    main(config)