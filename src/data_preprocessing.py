import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import jieba
import re
import yaml
from typing import Dict, List, Tuple
import os

class FraudDialogueDataset(Dataset):
    """欺诈对话数据集类"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载停用词
        self.stopwords = self._load_stopwords()
        
    def _load_stopwords(self):
        """加载停用词表"""
        stopwords = set()
        # 这里可以添加中文停用词文件
        stopwords_file = "stopwords.txt"
        if os.path.exists(stopwords_file):
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    stopwords.add(line.strip())
        return stopwords
    
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载训练和测试数据"""
        train_df = pd.read_csv(train_path, encoding='utf-8')
        test_df = pd.read_csv(test_path, encoding='utf-8')
        
        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")
        print(f"欺诈样本比例 - 训练集: {train_df['is_fraud'].mean():.2%}")
        print(f"欺诈样本比例 - 测试集: {test_df['is_fraud'].mean():.2%}")
        
        return train_df, test_df
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        # 去除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', str(text))
        # 分词
        words = jieba.lcut(text)
        # 去除停用词
        words = [word for word in words if word not in self.stopwords]
        
        return ' '.join(words)
    
    def prepare_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                     text_col: str = 'specific_dialogue_content',
                     label_col: str = 'is_fraud') -> Dict:
        """准备数据集"""
        
        # 1. 预处理文本
        train_df['processed_text'] = train_df[text_col].apply(self.preprocess_text)
        test_df['processed_text'] = test_df[text_col].apply(self.preprocess_text)
        
        # 2. 处理缺失值和类型转换
        # 填充缺失的标签
        if label_col in train_df.columns:
            train_df[label_col] = train_df[label_col].fillna(0)  # 缺失值填充为0（非欺诈）
        if label_col in test_df.columns:
            test_df[label_col] = test_df[label_col].fillna(0)
        
        # 确保标签是整数类型
        try:
            y_train = train_df[label_col].astype(int).tolist()
            y_test = test_df[label_col].astype(int).tolist()
        except Exception as e:
            print(f"标签转换错误: {e}")
            print("尝试修复标签...")
            
            # 尝试多种转换方式
            def safe_convert(x):
                try:
                    if pd.isna(x):
                        return 0
                    if isinstance(x, str):
                        if x.lower() in ['true', '是', '欺诈', 'fraud', '1']:
                            return 1
                        else:
                            return 0
                    return int(float(x))
                except:
                    return 0
            
            y_train = train_df[label_col].apply(safe_convert).tolist()
            y_test = test_df[label_col].apply(safe_convert).tolist()
        
        X_train = train_df['processed_text'].tolist()
        X_test = test_df['processed_text'].tolist()
        
        # 3. 统计信息
        print(f"\n数据统计:")
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        print(f"训练集欺诈比例: {sum(y_train)/len(y_train):.2%}")
        print(f"测试集欺诈比例: {sum(y_test)/len(y_test):.2%}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_df': train_df,
            'test_df': test_df
        }
    
    def create_dataloaders(self, X_train, y_train, X_test, y_test, 
                          tokenizer, batch_size=32, model_type='bert'):
        """创建数据加载器"""
        max_length = self.config['models']['bert']['max_length'] if model_type == 'bert' else self.config['models']['bilstm']['max_length']
        
        train_dataset = FraudDialogueDataset(X_train, y_train, tokenizer, max_length)
        test_dataset = FraudDialogueDataset(X_test, y_test, tokenizer, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader