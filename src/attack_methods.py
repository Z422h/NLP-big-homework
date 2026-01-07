# src/attack_methods.py - 修改同义词替换部分
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import jieba
from textattack import Attack
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import ModelWrapper
import warnings
warnings.filterwarnings('ignore')

# 尝试导入synonyms，如果失败则使用备用方法
try:
    import synonyms
    SYNONYMS_AVAILABLE = True
except ImportError:
    SYNONYMS_AVAILABLE = False
    print("警告: synonyms库不可用，使用简单同义词表")

class TextAttackWrapper(ModelWrapper):
    """TextAttack模型包装器"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def __call__(self, text_input_list):
        # 将文本转换为模型输入
        inputs = self.tokenizer(
            text_input_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # 转换为numpy数组
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        
        return logits

class SimpleSynonyms:
    """简单同义词查找器"""
    
    def __init__(self):
        # 简单的同义词映射表（中文）
        self.synonym_dict = {
            '你好': ['您好', '你们好', '大家好'],
            '谢谢': ['感谢', '多谢', '谢谢您'],
            '请问': ['请教', '问一下', '咨询'],
            '银行': ['金融机构', '储蓄所', '钱庄'],
            '客服': ['服务人员', '接待员', '服务员'],
            '密码': ['口令', '暗号', '密钥'],
            '验证码': ['验证数字', '确认码', '安全码'],
            '转账': ['汇款', '划账', '转款'],
            '账户': ['户头', '账号', '存款账户'],
            '安全': ['保险', '可靠', '安稳'],
            '立即': ['马上', '立刻', '即刻'],
            '提供': ['供给', '给予', '供应'],
            '需要': ['要求', '需求', '必要'],
            '请': ['恳请', '请求', '要求'],
            '我': ['本人', '俺', '咱'],
            '你': ['您', '阁下', '贵方'],
            '他': ['她', '它', '对方'],
            '的': ['之', '地', '得'],
            '了': ['啦', '咯', '罢'],
            '在': ['于', '正在', '处在']
        }
    
    def nearby(self, word):
        """模拟synonyms.nearby接口"""
        if word in self.synonym_dict:
            return [word] + self.synonym_dict[word], [1.0] + [0.8] * len(self.synonym_dict[word])
        else:
            # 如果没有找到同义词，返回原词
            return [word], [1.0]

class AdversarialAttack:
    """对抗攻击生成器"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 初始化同义词库
        if SYNONYMS_AVAILABLE:
            self.synonyms_lib = synonyms
        else:
            self.synonyms_lib = SimpleSynonyms()
            print("使用简单同义词表进行攻击")
    
    def textfooler_attack(self, texts: List[str], labels: List[int], 
                         num_examples: int = 100) -> Tuple[List[str], List[int], List[float]]:
        """使用TextFooler进行攻击"""
        
        model_wrapper = TextAttackWrapper(self.model, self.tokenizer, self.device)
        
        # 创建攻击
        attack = TextFoolerJin2019.build(model_wrapper)
        
        attacked_texts = []
        original_labels = []
        success_rates = []
        
        for i, (text, label) in enumerate(zip(texts[:num_examples], labels[:num_examples])):
            try:
                # 创建攻击样本
                attack_result = attack.attack(text, label)
                
                if attack_result.success:
                    attacked_texts.append(attack_result.perturbed_text)
                    original_labels.append(label)
                    success_rates.append(1.0)
                else:
                    attacked_texts.append(text)  # 攻击失败，使用原始文本
                    original_labels.append(label)
                    success_rates.append(0.0)
                    
            except Exception as e:
                print(f"Error attacking sample {i}: {e}")
                attacked_texts.append(text)
                original_labels.append(label)
                success_rates.append(0.0)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{min(num_examples, len(texts))} samples")
        
        avg_success_rate = np.mean(success_rates)
        print(f"TextFooler attack completed. Success rate: {avg_success_rate:.2%}")
        
        return attacked_texts, original_labels, success_rates
    
    def synonym_replacement_attack(self, texts: List[str], labels: List[int], 
                                  replacement_rate: float = 0.2) -> Tuple[List[str], List[int]]:
        """同义词替换攻击"""
        
        attacked_texts = []
        
        for text_idx, text in enumerate(texts):
            words = jieba.lcut(text)
            num_to_replace = max(1, int(len(words) * replacement_rate))
            
            # 随机选择要替换的单词
            if len(words) > 0:
                replace_indices = np.random.choice(len(words), 
                                                  min(num_to_replace, len(words)), 
                                                  replace=False)
            else:
                replace_indices = []
            
            new_words = words.copy()
            for idx in replace_indices:
                word = words[idx]
                
                # 获取同义词
                try:
                    syns = self.synonyms_lib.nearby(word)[0]
                    if len(syns) > 1:
                        # 选择最相似但不是自身的同义词
                        for syn in syns[1:]:  # 跳过第一个（通常是自身）
                            if syn != word and len(syn) > 0:
                                new_words[idx] = syn
                                break
                except Exception as e:
                    # 如果找不到同义词，保持原词
                    pass
            
            attacked_text = ''.join(new_words)
            attacked_texts.append(attacked_text)
            
            if (text_idx + 1) % 50 == 0:
                print(f"已处理 {text_idx + 1}/{len(texts)} 个文本")
        
        return attacked_texts, labels
    
    def create_adversarial_dataset(self, texts: List[str], labels: List[int], 
                                  attack_type: str = 'textfooler', 
                                  num_samples: int = None) -> Dict[str, Any]:
        """创建对抗性数据集"""
        
        if num_samples is None:
            num_samples = len(texts)
        
        print(f"Creating adversarial dataset with {attack_type} attack...")
        print(f"Number of samples to attack: {num_samples}")
        
        if attack_type == 'textfooler':
            attacked_texts, attacked_labels, success_rates = self.textfooler_attack(
                texts, labels, num_samples
            )
            
        elif attack_type == 'synonym':
            attacked_texts, attacked_labels = self.synonym_replacement_attack(
                texts[:num_samples], labels[:num_samples]
            )
            success_rates = [1.0] * len(attacked_texts)  # 假设全部成功
            
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")
        
        # 计算攻击成功率
        if attack_type == 'textfooler':
            success_rate = np.mean(success_rates)
        else:
            # 对于同义词替换，我们需要评估模型在攻击样本上的性能
            success_rate = None  # 将在实验部分计算
        
        return {
            'texts': attacked_texts,
            'labels': attacked_labels,
            'original_texts': texts[:num_samples],
            'original_labels': labels[:num_samples],
            'success_rates': success_rates,
            'attack_type': attack_type
        }
    
    def evaluate_attack_success(self, original_preds: List[int], 
                               adversarial_preds: List[int], 
                               true_labels: List[int]) -> Dict[str, float]:
        """评估攻击成功率"""
        
        original_correct = sum(1 for p, t in zip(original_preds, true_labels) if p == t)
        adversarial_correct = sum(1 for p, t in zip(adversarial_preds, true_labels) if p == t)
        
        # 攻击成功：原本正确但攻击后错误
        attack_success = 0
        total_attackable = 0
        
        for op, ap, tl in zip(original_preds, adversarial_preds, true_labels):
            if op == tl:  # 原本分类正确
                total_attackable += 1
                if ap != tl:  # 攻击后分类错误
                    attack_success += 1
        
        attack_success_rate = attack_success / total_attackable if total_attackable > 0 else 0
        
        return {
            'original_accuracy': original_correct / len(true_labels),
            'adversarial_accuracy': adversarial_correct / len(true_labels),
            'attack_success_rate': attack_success_rate,
            'accuracy_drop': (original_correct - adversarial_correct) / len(true_labels)
        }