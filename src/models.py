import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW  # 从 torch 导入
from typing import Dict, Any, Optional
import numpy as np

class BiLSTMClassifier(nn.Module):
    """BiLSTM分类器"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300, 
                 hidden_dim: int = 256, num_layers: int = 2, 
                 dropout: float = 0.5, num_classes: int = 2):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        
        # 应用注意力掩码
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            embedded = embedded * attention_mask.unsqueeze(-1)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.dropout(hidden)
        logits = self.fc(output)
        
        return logits

class BERTClassifier(nn.Module):
    def __init__(self, model_name: str = 'bert-base-chinese', num_classes: int = 2, dropout: float = 0.3):
        super(BERTClassifier, self).__init__()
        
        # 从本地路径加载
        local_model_path = "./models/bert-base-chinese"
        
        try:
            print(f"从 {local_model_path} 加载BERT模型...")
            
            # 禁用SSL验证（针对证书问题）
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # 检查是否存在safetensors文件
            safetensors_path = os.path.join(local_model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                print(f"检测到safetensors文件，使用safetensors加载...")
                self.bert = BertModel.from_pretrained(local_model_path, local_files_only=True)
            else:
                self.bert = BertModel.from_pretrained(local_model_path, local_files_only=True)
            
            print("✓ BERT模型加载成功")
            
        except Exception as e:
            print(f"加载BERT模型失败: {e}")
            print("使用简化版本的BERT...")
            self.bert = None
            self.embedding = nn.Embedding(21128, 768)
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072)
                for _ in range(6)
            ])
            self.pooler = nn.Linear(768, 768)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        if self.bert is not None:
            # 使用真实的BERT
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_output = outputs.pooler_output
        else:
            # 使用简化版本
            embedded = self.embedding(input_ids)
            x = embedded
            for layer in self.transformer_layers:
                x = layer(x)
            pooled_output = x.mean(dim=1)
            pooled_output = self.pooler(pooled_output)
            pooled_output = torch.tanh(pooled_output)
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class ModelManager:
    """模型管理器"""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        self.models = {}
        
    def initialize_model(self, model_type: str, vocab_size: Optional[int] = None, num_classes: int = 2):
        """初始化模型"""
        if model_type == 'bert':
            model = BERTClassifier(
                model_name=self.config['models']['bert']['model_name'],
                num_classes=num_classes
            )
            
            # 尝试加载tokenizer
            local_model_path = "./models/bert-base-chinese"
            if os.path.exists(local_model_path):
                try:
                    # 禁用SSL验证
                    import ssl
                    ssl._create_default_https_context = ssl._create_unverified_context
                    
                    tokenizer = BertTokenizer.from_pretrained(local_model_path, local_files_only=True)
                    print(f"✓ 成功从 {local_model_path} 加载tokenizer")
                except Exception as e:
                    print(f"无法加载tokenizer: {e}")
                    print("使用简单分词...")
                    tokenizer = None
            else:
                print("警告: BERT模型目录不存在")
                tokenizer = None
                
        elif model_type == 'bilstm':
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for BiLSTM model")
            
            model = BiLSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=self.config['models']['bilstm']['embedding_dim'],
                hidden_dim=self.config['models']['bilstm']['hidden_dim'],
                num_layers=self.config['models']['bilstm']['num_layers'],
                dropout=self.config['models']['bilstm']['dropout'],
                num_classes=num_classes
            )
            # 对于BiLSTM，我们使用简单的分词器
            tokenizer = None  # 需要在数据预处理时构建词汇表
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.to(self.device)
        self.models[model_type] = model
        
        return model, tokenizer
    
    def train_epoch(self, model, train_loader, optimizer, criterion, scheduler=None):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def evaluate(self, model, test_loader, criterion=None):
        """评估模型"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                
                if criterion:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader) if criterion else None
        
        return accuracy, avg_loss, all_preds, all_labels
    
    def save_model(self, model, path: str):
        """保存模型"""
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, model, path: str):
        """加载模型"""
        model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
        return model