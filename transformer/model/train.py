# trainer.py

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, IterableDataset
import tqdm
import os
import json
import numpy as np
from typing import Optional, Dict, Any

# 尝试导入 wandb，如果未安装则跳过
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

class Trainer:
    """一个通用的 PyTorch 模型训练器。

    该类封装了标准的训练和评估循环、模型保存、日志记录等功能。
    """
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 lr_scheduler: Optional[_LRScheduler] = None,
                 wandb_run: Optional[Any] = None,
                 grad_clip_val: Optional[float] = 1.0,
                 save_dir: str = "checkpoints"):
        """
        初始化 Trainer。

        Args:
            model (nn.Module): 需要训练的 PyTorch 模型。
            optimizer (Optimizer): 优化器 (例如, AdamW)。
            criterion (nn.Module): 损失函数 (例如, CrossEntropyLoss)。
            device (torch.device): 运行模型的设备 ('cuda' or 'cpu')。
            lr_scheduler (_LRScheduler, optional): 学习率调度器。默认为 None。
            wandb_run (Any, optional): W&B 的运行实例，用于日志记录。默认为 None。
            grad_clip_val (float, optional): 梯度裁剪的值。如果为 None，则不进行裁剪。默认为 1.0。
            save_dir (str, optional): 保存最佳模型的目录。默认为 "checkpoints"。
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.wandb_run = wandb_run
        self.grad_clip_val = grad_clip_val
        self.save_dir = save_dir
        
        self.best_val_loss = float('inf')
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, "best_model.pt")

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """执行一个完整的训练轮次 (epoch)。"""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm.tqdm(train_loader, desc=f"Training Epoch"):
            # 假设 batch 是一个张量或元组/列表
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else: # 适用于我们的流式语言模型数据
                inputs = batch[:, :-1].to(self.device)
                targets = batch[:, 1:].to(self.device)

            # --- 前向传播 ---
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # --- 计算损失 ---
            loss = self.criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            
            # --- 反向传播 ---
            loss.backward()
            
            # --- 梯度裁剪 ---
            if self.grad_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
            
            # --- 更新权重 ---
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader) if len(train_loader) > 0 else total_loss

    def _evaluate(self, val_loader: DataLoader) -> float:
        """在验证集上评估模型。"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc=f"Evaluating"):
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs = batch[:, :-1].to(self.device)
                    targets = batch[:, 1:].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader) if len(val_loader) > 0 else total_loss

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """保存模型检查点。"""
        print(f"Validation loss decreased ({self.best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
        self.best_val_loss = val_loss
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        if self.lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
            
        torch.save(checkpoint, self.save_path)

    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float):
        """记录指标到 W&B。"""
        if self.wandb_run and WANDB_AVAILABLE:
            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": self.best_val_loss,
            }
            if self.lr_scheduler:
                log_data["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                
            self.wandb_run.log(log_data)
            print(f"Metrics logged to W&B for epoch {epoch}.")

    def train(self, num_epochs: int, train_loader: DataLoader, val_loader: DataLoader):
        """
        启动完整的训练流程。

        Args:
            num_epochs (int): 训练的总轮次数。
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
        """
        print("--- Starting Training ---")
        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._evaluate(val_loader)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}\n")
            
            # 更新学习率调度器（如果存在）
            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss) # Step on validation loss for ReduceLROnPlateau
            
            # 记录指标
            self._log_metrics(epoch, train_loss, val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self._save_checkpoint(epoch, val_loss)
        
        print("--- Training Finished ---")
        print(f"Best model saved at: {self.save_path} with validation loss: {self.best_val_loss:.4f}")

# 在 train.py 中
class StreamingQADataset(IterableDataset):
    def __init__(self, file_path, tokenizer, seq_len, split='train', split_ratio=0.9):
        """
        一个集成了所有功能的流式数据集。
        - file_path: 数据文件路径
        - tokenizer: 分词器实例
        - seq_len: 模型序列长度
        - split: 'train' 或 'val'，决定这个实例属于哪个数据集
        - split_ratio: 训练集所占的比例
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.split_ratio = split_ratio
        self.token_buffer = [] # 用于累积 token 的缓冲区
        self.stride = 1 # 滑动窗口的步长

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # enumerate data.items() 来决定一个样本属于训练集还是验证集
        for i, (key, value) in enumerate(data.items()):
            # 根据索引和比例决定是否处理当前数据条目
            is_train_sample = (i / len(data)) < self.split_ratio
            if self.split == 'train' and not is_train_sample:
                continue
            if self.split == 'val' and is_train_sample:
                continue

            # 1. 格式化文本
            q = f"QUESTION: {value['QUESTION']}"
            c = f"CONTEXT: {' '.join(value['CONTEXTS'])}"
            la = f"LONG_ANSWER: {value['LONG_ANSWER']}"
            record_text = " ".join([q, c, la]) + f" {self.tokenizer.EOS_ID} {self.tokenizer.SOS_ID} "
            
            # 2. 实时编码并放入缓冲区
            new_tokens = self.tokenizer.encode_corpus(record_text)
            self.token_buffer.extend(new_tokens)
            
            # 3. 从缓冲区中生成固定长度的样本
            while len(self.token_buffer) >= self.seq_len:
                # 取前 seq_len 个 token 作为一个样本
                chunk = self.token_buffer[:self.seq_len]
                yield torch.tensor(chunk, dtype=torch.long)
                
                # 滑动窗口
                self.token_buffer = self.token_buffer[self.stride:]