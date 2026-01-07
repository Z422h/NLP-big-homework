import torch
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import jieba
import synonyms
from tqdm import tqdm

class FPPDefense:
    """FPP（以扰动对抗扰动）防御框架"""
    
    def __init__(self, base_classifier, tokenizer, training_texts: List[str], 
                 config: Dict[str, Any], device: str = 'cuda'):
        self.base_classifier = base_classifier
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # 从训练数据构建n-gram频率统计
        self.word_freq, self.bigram_freq = self._build_frequency_stats(training_texts)
        
        # FPP参数
        self.lambda_param = config['fpp']['lambda_param']
        self.kappa = config['fpp']['kappa']
        self.num_samples = config['fpp']['num_samples']
        self.voting_type = config['fpp']['voting_type']
        
    def _build_frequency_stats(self, texts: List[str]) -> Tuple[Counter, Dict[str, Counter]]:
        """构建1-gram和2-gram频率统计"""
        word_counter = Counter()
        bigram_counter = defaultdict(Counter)
        
        for text in tqdm(texts, desc="Building frequency statistics"):
            words = jieba.lcut(text)
            
            # 统计1-gram
            for word in words:
                word_counter[word] += 1
            
            # 统计2-gram
            for i in range(len(words) - 1):
                bigram_counter[words[i]][words[i + 1]] += 1
        
        return word_counter, bigram_counter
    
    def _get_candidate_words(self, word: str) -> List[str]:
        """获取候选替换词"""
        candidates = []
        
        # 获取同义词
        try:
            syns = synonyms.nearby(word)[0]
            candidates.extend(syns[:5])  # 取前5个最相似的
        except:
            pass
        
        # 添加原词
        if word not in candidates:
            candidates.append(word)
        
        return list(set(candidates))  # 去重
    
    def _calculate_synthesized_frequency(self, word: str, next_word: str = None) -> float:
        """计算综合频率"""
        
        candidates = self._get_candidate_words(word)
        
        if not candidates:
            return 0.0
        
        # 计算1-gram相对频率
        total_freq = sum(self.word_freq.get(w, 1) for w in candidates)  # 加1平滑
        word_freq = self.word_freq.get(word, 1)
        p1 = word_freq / total_freq if total_freq > 0 else 0
        
        # 计算2-gram相对频率（如果提供了下一个词）
        p2 = 0.0
        if next_word and word in self.bigram_freq:
            bigram_counts = self.bigram_freq[word]
            total_bigram_freq = sum(bigram_counts.values())
            if total_bigram_freq > 0:
                p2 = bigram_counts.get(next_word, 1) / total_bigram_freq
        
        # 综合频率
        p12 = (1 - self.lambda_param) * p1 + self.lambda_param * p2
        
        return p12
    
    def _perturb_input(self, text: str) -> str:
        """第一阶段：基于n-gram频率的输入扰动"""
        words = jieba.lcut(text)
        n = len(words)
        max_replacements = int(n * self.kappa)
        
        # 计算每个位置的替换概率
        replace_probs = []
        for i in range(n):
            word = words[i]
            next_word = words[i + 1] if i < n - 1 else None
            
            # 获取候选词
            candidates = self._get_candidate_words(word)
            if len(candidates) <= 1:
                replace_probs.append(0.0)
                continue
            
            # 计算原词的频率
            p_original = self._calculate_synthesized_frequency(word, next_word)
            
            # 找到最佳替换词
            best_candidate = word
            best_freq = p_original
            
            for candidate in candidates:
                if candidate == word:
                    continue
                p_candidate = self._calculate_synthesized_frequency(candidate, next_word)
                if p_candidate > best_freq:
                    best_freq = p_candidate
                    best_candidate = candidate
            
            # 计算替换概率
            delta = best_freq - p_original
            replace_probs.append(max(0, delta))  # 确保概率非负
        
        # 随机决定是否替换
        new_words = words.copy()
        replacements = 0
        
        for i in range(n):
            if replacements >= max_replacements:
                break
            
            if np.random.random() < replace_probs[i]:
                word = words[i]
                next_word = words[i + 1] if i < n - 1 else None
                
                candidates = self._get_candidate_words(word)
                if len(candidates) <= 1:
                    continue
                
                # 选择最佳候选词
                best_candidate = word
                best_freq = self._calculate_synthesized_frequency(word, next_word)
                
                for candidate in candidates:
                    if candidate == word:
                        continue
                    p_candidate = self._calculate_synthesized_frequency(candidate, next_word)
                    if p_candidate > best_freq:
                        best_freq = p_candidate
                        best_candidate = candidate
                
                if best_candidate != word:
                    new_words[i] = best_candidate
                    replacements += 1
        
        return ''.join(new_words)
    
    def _random_perturbation(self, text: str) -> str:
        """生成随机扰动版本"""
        words = jieba.lcut(text)
        n = len(words)
        max_replacements = int(n * self.kappa)
        
        if max_replacements == 0:
            return text
        
        # 随机选择要替换的位置
        num_replacements = np.random.randint(1, max_replacements + 1)
        replace_indices = np.random.choice(n, num_replacements, replace=False)
        
        new_words = words.copy()
        for idx in replace_indices:
            word = words[idx]
            candidates = self._get_candidate_words(word)
            
            if len(candidates) > 1:
                # 随机选择一个不是原词的候选词
                candidate = np.random.choice([c for c in candidates if c != word])
                new_words[idx] = candidate
        
        return ''.join(new_words)
    
    def _majority_vote(self, predictions: List[int]) -> int:
        """多数投票"""
        from collections import Counter
        pred_counter = Counter(predictions)
        most_common = pred_counter.most_common(1)[0]
        
        if most_common[1] > len(predictions) / 2:
            return most_common[0]
        else:
            return -1  # 拒绝预测
    
    def _plurality_vote(self, predictions: List[int]) -> int:
        """复数投票"""
        from collections import Counter
        pred_counter = Counter(predictions)
        return pred_counter.most_common(1)[0][0]
    
    def predict(self, text: str) -> Tuple[int, List[int]]:
        """FPP增强预测"""
        
        # 第一阶段：输入扰动
        perturbed_text = self._perturb_input(text)
        
        # 第二阶段：随机扰动集成
        predictions = []
        
        for _ in range(self.num_samples):
            # 生成随机扰动版本
            random_perturbed = self._random_perturbation(perturbed_text)
            
            # 基分类器预测
            inputs = self.tokenizer(
                random_perturbed,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            self.base_classifier.eval()
            with torch.no_grad():
                outputs = self.base_classifier(**inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                pred = torch.argmax(logits, dim=1).item()
                predictions.append(pred)
        
        # 投票
        if self.voting_type == 'majority':
            final_pred = self._majority_vote(predictions)
        else:  # plurality
            final_pred = self._plurality_vote(predictions)
        
        return final_pred, predictions
    
    def evaluate_fpp(self, texts: List[str], labels: List[int], 
                    batch_size: int = 32) -> Dict[str, Any]:
        """评估FPP防御效果"""
        
        base_predictions = []
        fpp_predictions = []
        fpp_all_votes = []
        
        print(f"Evaluating FPP defense on {len(texts)} samples...")
        
        # 首先获取基分类器的预测
        self.base_classifier.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Base classifier predictions"):
                batch_texts = texts[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.base_classifier(**inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                base_predictions.extend(batch_preds)
        
        # 然后获取FPP增强预测
        for i in tqdm(range(len(texts)), desc="FPP predictions"):
            text = texts[i]
            fpp_pred, all_votes = self.predict(text)
            
            fpp_predictions.append(fpp_pred)
            fpp_all_votes.append(all_votes)
        
        # 计算指标
        base_correct = sum(1 for p, l in zip(base_predictions, labels) if p == l)
        base_accuracy = base_correct / len(labels)
        
        # 对于FPP，需要处理拒绝预测的情况
        fpp_correct = 0
        fpp_total = 0
        rejected = 0
        
        for fpp_pred, label in zip(fpp_predictions, labels):
            if fpp_pred == -1:  # 拒绝预测
                rejected += 1
            else:
                fpp_total += 1
                if fpp_pred == label:
                    fpp_correct += 1
        
        fpp_accuracy = fpp_correct / fpp_total if fpp_total > 0 else 0
        rejection_rate = rejected / len(labels)
        
        return {
            'base_accuracy': base_accuracy,
            'fpp_accuracy': fpp_accuracy,
            'rejection_rate': rejection_rate,
            'base_predictions': base_predictions,
            'fpp_predictions': fpp_predictions,
            'fpp_votes': fpp_all_votes
        }