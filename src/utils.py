import torch
import numpy as np
import random
import os
import json
import yaml
from typing import Any, Dict, List, Union
import logging

def setup_logging(log_file: str = "experiment.log"):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_json(data: Any, filepath: str):
    """保存JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: str) -> Any:
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_yaml(data: Dict, filepath: str):
    """保存YAML文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

def load_yaml(filepath: str) -> Dict:
    """加载YAML文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def calculate_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """计算分类指标"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def analyze_fraud_patterns(texts: List[str], labels: List[int], 
                          predictions: List[int]) -> Dict[str, Any]:
    """分析欺诈模式"""
    
    # 识别误分类的样本
    misclassified = []
    for i, (pred, label, text) in enumerate(zip(predictions, labels, texts)):
        if pred != label:
            misclassified.append({
                'text': text,
                'true_label': label,
                'predicted_label': pred
            })
    
    # 分析欺诈话术特征
    fraud_keywords = ['转账', '验证码', '安全账户', '公安局', '客服', '退款', 
                     '密码', '身份证', '银行卡', '汇款', '贷款', '投资']
    
    keyword_counts = {keyword: 0 for keyword in fraud_keywords}
    for text in texts:
        for keyword in fraud_keywords:
            if keyword in text:
                keyword_counts[keyword] += 1
    
    return {
        'misclassified_count': len(misclassified),
        'misclassified_samples': misclassified[:10],  # 只保留前10个
        'keyword_distribution': keyword_counts,
        'error_rate': len(misclassified) / len(texts)
    }