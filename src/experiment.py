import jieba
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import yaml
import json
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['experiment']['device'] 
                                  if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建输出目录
        self._create_output_dirs()
        
        # 实验结果存储
        self.results = {}
        
    def _create_output_dirs(self):
        """创建输出目录"""
        paths = self.config['paths']
        for dir_path in paths.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def run_baseline_experiment(self, data_manager, model_manager, 
                           model_type: str = 'bert') -> Dict[str, Any]:
        """运行基线实验"""
        print(f"\n{'='*60}")
        print(f"Running baseline experiment with {model_type.upper()}")
        print('='*60)
        
        # 加载数据
        from src.data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        
        train_df, test_df = preprocessor.load_data(
            os.path.join(self.config['paths']['data_dir'], '训练集结果.csv'),
            os.path.join(self.config['paths']['data_dir'], '测试集结果.csv')
        )
        
        # 准备数据集
        data_dict = preprocessor.prepare_datasets(train_df, test_df)
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        # 初始化模型和tokenizer
        if model_type == 'bert':
            model, tokenizer = model_manager.initialize_model('bert', num_classes=2)
        else:  # bilstm
            # 构建词汇表
            from collections import Counter
            all_words = []
            for text in X_train:
                all_words.extend(jieba.lcut(text))
            word_counter = Counter(all_words)
            vocab = {word: idx+1 for idx, (word, _) in enumerate(word_counter.most_common(10000))}
            vocab['<PAD>'] = 0
            vocab['<UNK>'] = len(vocab)
            
            model, tokenizer = model_manager.initialize_model('bilstm', 
                                                            vocab_size=len(vocab), 
                                                            num_classes=2)
            # 为BiLSTM创建自定义tokenizer
            class BiLSTMTokenizer:
                def __init__(self, vocab):
                    self.vocab = vocab
                    self.vocab_inv = {v: k for k, v in vocab.items()}
                
                def __call__(self, text, **kwargs):
                    words = jieba.lcut(text)
                    ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
                    return {'input_ids': torch.tensor(ids)}
            
            tokenizer = BiLSTMTokenizer(vocab)
        
        # 创建数据加载器
        train_loader, test_loader = preprocessor.create_dataloaders(
            X_train, y_train, X_test, y_test,
            tokenizer,
            batch_size=int(self.config['experiment']['batch_size']),  # 确保是整数
            model_type=model_type
        )
        
        # 训练模型
        print(f"\nTraining {model_type.upper()} model...")
        criterion = torch.nn.CrossEntropyLoss()
        
        if model_type == 'bert':
            # 确保学习率是浮点数
            lr = self.config['models']['bert']['learning_rate']
            if isinstance(lr, str):
                # 处理科学计数法
                if 'e' in lr:
                    base, exp = lr.split('e')
                    lr = float(base) * (10 ** int(exp))
                else:
                    lr = float(lr)
            
            optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=lr)
        else:
            # 确保学习率是浮点数
            lr = self.config['models']['bilstm']['learning_rate']
            if isinstance(lr, str):
                if 'e' in lr:
                    base, exp = lr.split('e')
                    lr = float(base) * (10 ** int(exp))
                else:
                    lr = float(lr)
            
            optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=lr)
        
        # 训练循环
        best_accuracy = 0
        for epoch in range(int(self.config['experiment']['num_epochs'])):  # 确保是整数
            train_loss, train_acc = model_manager.train_epoch(
                model, train_loader, optimizer, criterion
            )
            
            test_acc, test_loss, test_preds, test_labels = model_manager.evaluate(
                model, test_loader, criterion
            )
            
            print(f"Epoch {epoch+1}/{self.config['experiment']['num_epochs']}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                # 保存最佳模型
                model_path = os.path.join(
                    self.config['paths']['model_dir'], 
                    f'{model_type}_best_model.pth'
                )
                model_manager.save_model(model, model_path)
        
        # 最终评估
        test_acc, _, test_preds, test_labels = model_manager.evaluate(model, test_loader)
        
        # 保存结果
        result = {
            'model_type': model_type,
            'test_accuracy': test_acc,
            'test_predictions': test_preds,
            'test_labels': test_labels,
            'model': model,
            'tokenizer': tokenizer
        }
        
        # 打印分类报告
        print(f"\nClassification Report for {model_type.upper()}:")
        print(classification_report(test_labels, test_preds, target_names=['Non-Fraud', 'Fraud']))
        
        # 保存到实验结果
        self.results[f'baseline_{model_type}'] = result
        
        return result
    
    def run_adversarial_attack_experiment(self, model, tokenizer, texts: List[str], 
                                         labels: List[int], attack_type: str = 'textfooler',
                                         num_samples: int = 100) -> Dict[str, Any]:
        """运行对抗攻击实验"""
        print(f"\n{'='*60}")
        print(f"Running {attack_type.upper()} attack experiment")
        print('='*60)
        
        from src.attack_methods import AdversarialAttack
        
        attack_generator = AdversarialAttack(model, tokenizer, self.device)
        
        # 生成对抗样本
        adversarial_data = attack_generator.create_adversarial_dataset(
            texts, labels, attack_type, num_samples
        )
        
        # 评估攻击效果
        # 首先获取原始预测
        original_preds = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(adversarial_data['original_texts']), 32):
                batch_texts = adversarial_data['original_texts'][i:i+32]
                
                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = model(**inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                original_preds.extend(batch_preds)
        
        # 获取对抗样本的预测
        adversarial_preds = []
        with torch.no_grad():
            for i in range(0, len(adversarial_data['texts']), 32):
                batch_texts = adversarial_data['texts'][i:i+32]
                
                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = model(**inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                adversarial_preds.extend(batch_preds)
        
        # 评估攻击成功率
        attack_metrics = attack_generator.evaluate_attack_success(
            original_preds, adversarial_preds, adversarial_data['labels']
        )
        
        print(f"\nAttack Metrics:")
        for metric, value in attack_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        result = {
            'attack_type': attack_type,
            'adversarial_data': adversarial_data,
            'original_predictions': original_preds,
            'adversarial_predictions': adversarial_preds,
            'attack_metrics': attack_metrics
        }
        
        # 保存对抗样本
        adversarial_samples_path = os.path.join(
            self.config['paths']['adversarial_dir'],
            f'adversarial_samples_{attack_type}.json'
        )
        
        with open(adversarial_samples_path, 'w', encoding='utf-8') as f:
            # 只保存前100个样本用于查看
            samples_to_save = []
            for i in range(min(100, len(adversarial_data['texts']))):
                samples_to_save.append({
                    'original_text': adversarial_data['original_texts'][i],
                    'adversarial_text': adversarial_data['texts'][i],
                    'original_label': int(adversarial_data['original_labels'][i]),
                    'original_pred': int(original_preds[i]),
                    'adversarial_pred': int(adversarial_preds[i])
                })
            
            json.dump(samples_to_save, f, ensure_ascii=False, indent=2)
        
        print(f"Adversarial samples saved to {adversarial_samples_path}")
        
        # 保存到实验结果
        self.results[f'attack_{attack_type}'] = result
        
        return result
    
    def run_fpp_defense_experiment(self, model, tokenizer, training_texts: List[str],
                                  test_texts: List[str], test_labels: List[int]) -> Dict[str, Any]:
        """运行FPP防御实验"""
        print(f"\n{'='*60}")
        print("Running FPP defense experiment")
        print('='*60)
        
        from src.fpp_defense import FPPDefense
        
        # 初始化FPP防御
        fpp_defense = FPPDefense(
            model, tokenizer, training_texts, self.config, self.device
        )
        
        # 评估FPP防御效果
        fpp_results = fpp_defense.evaluate_fpp(test_texts, test_labels)
        
        print(f"\nFPP Defense Results:")
        print(f"  Base Model Accuracy: {fpp_results['base_accuracy']:.4f}")
        print(f"  FPP Enhanced Accuracy: {fpp_results['fpp_accuracy']:.4f}")
        print(f"  Rejection Rate: {fpp_results['rejection_rate']:.4f}")
        
        result = {
            'fpp_results': fpp_results,
            'fpp_defense': fpp_defense
        }
        
        # 保存到实验结果
        self.results['fpp_defense'] = result
        
        return result
    
    def run_ablation_study(self, model, tokenizer, texts: List[str], 
                          labels: List[int]) -> Dict[str, Any]:
        """运行消融实验"""
        print(f"\n{'='*60}")
        print("Running ablation study")
        print('='*60)
        
        from src.attack_methods import AdversarialAttack
        
        attack_generator = AdversarialAttack(model, tokenizer, self.device)
        
        # 实验1：只替换同义词 vs 改写整句
        # 同义词替换攻击
        synonym_texts, synonym_labels = attack_generator.synonym_replacement_attack(
            texts[:200], labels[:200], replacement_rate=0.2
        )
        
        # 评估同义词替换效果
        model.eval()
        original_preds = []
        synonym_preds = []
        
        with torch.no_grad():
            # 原始预测
            for i in range(0, len(texts[:200]), 32):
                batch_texts = texts[i:i+32]
                
                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = model(**inputs)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                original_preds.extend(batch_preds)
            
            # 同义词替换预测
            for i in range(0, len(synonym_texts), 32):
                batch_texts = synonym_texts[i:i+32]
                
                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = model(**inputs)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                synonym_preds.extend(batch_preds)
        
        # 计算指标
        original_acc = accuracy_score(labels[:200], original_preds)
        synonym_acc = accuracy_score(synonym_labels, synonym_preds)
        
        print(f"\nAblation Study Results:")
        print(f"  Original Accuracy: {original_acc:.4f}")
        print(f"  Synonym Replacement Accuracy: {synonym_acc:.4f}")
        print(f"  Accuracy Drop: {original_acc - synonym_acc:.4f}")
        
        # 分析哪些改写最有效
        successful_attacks = []
        for i in range(len(original_preds)):
            if original_preds[i] == labels[i] and synonym_preds[i] != labels[i]:
                successful_attacks.append({
                    'original_text': texts[i],
                    'modified_text': synonym_texts[i],
                    'label': int(labels[i]),
                    'original_pred': int(original_preds[i]),
                    'modified_pred': int(synonym_preds[i])
                })
        
        print(f"  Number of successful attacks: {len(successful_attacks)}")
        
        # 保存成功攻击的示例
        if successful_attacks:
            ablation_path = os.path.join(
                self.config['paths']['result_dir'],
                'ablation_study_results.json'
            )
            
            with open(ablation_path, 'w', encoding='utf-8') as f:
                json.dump(successful_attacks[:20], f, ensure_ascii=False, indent=2)
            
            print(f"Successful attack examples saved to {ablation_path}")
        
        result = {
            'original_accuracy': original_acc,
            'synonym_accuracy': synonym_acc,
            'accuracy_drop': original_acc - synonym_acc,
            'successful_attacks': successful_attacks[:10]  # 只保存前10个
        }
        
        self.results['ablation_study'] = result
        
        return result
    
    def visualize_results(self):
        """可视化实验结果"""
        print(f"\n{'='*60}")
        print("Visualizing results")
        print('='*60)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 模型性能对比
        if 'baseline_bert' in self.results and 'baseline_bilstm' in self.results:
            models = ['BERT', 'BiLSTM']
            accuracies = [
                self.results['baseline_bert']['test_accuracy'],
                self.results['baseline_bilstm']['test_accuracy']
            ]
            
            axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightcoral'])
            axes[0, 0].set_title('Model Performance Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_ylim([0, 1])
            
            for i, acc in enumerate(accuracies):
                axes[0, 0].text(i, acc + 0.01, f'{acc:.4f}', ha='center')
        
        # 2. 攻击效果对比
        attack_results = []
        for key in self.results:
            if key.startswith('attack_'):
                attack_type = key.replace('attack_', '')
                metrics = self.results[key]['attack_metrics']
                attack_results.append({
                    'type': attack_type,
                    'original_acc': metrics['original_accuracy'],
                    'adversarial_acc': metrics['adversarial_accuracy'],
                    'attack_success': metrics['attack_success_rate']
                })
        
        if attack_results:
            attack_types = [r['type'] for r in attack_results]
            original_accs = [r['original_acc'] for r in attack_results]
            adversarial_accs = [r['adversarial_acc'] for r in attack_results]
            
            x = np.arange(len(attack_types))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, original_accs, width, label='Original', color='skyblue')
            axes[0, 1].bar(x + width/2, adversarial_accs, width, label='Adversarial', color='lightcoral')
            axes[0, 1].set_title('Attack Effectiveness Comparison')
            axes[0, 1].set_xlabel('Attack Type')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(attack_types)
            axes[0, 1].legend()
            axes[0, 1].set_ylim([0, 1])
        
        # 3. FPP防御效果
        if 'fpp_defense' in self.results:
            fpp_results = self.results['fpp_defense']['fpp_results']
            
            labels = ['Base Model', 'FPP Enhanced']
            values = [fpp_results['base_accuracy'], fpp_results['fpp_accuracy']]
            
            axes[1, 0].bar(labels, values, color=['skyblue', 'lightgreen'])
            axes[1, 0].set_title('FPP Defense Effectiveness')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_ylim([0, 1])
            
            for i, val in enumerate(values):
                axes[1, 0].text(i, val + 0.01, f'{val:.4f}', ha='center')
            
            # 添加拒绝率标注
            rejection_rate = fpp_results['rejection_rate']
            axes[1, 0].text(1, values[1] + 0.05, f'Rejection: {rejection_rate:.2%}', 
                           ha='center', fontsize=10)
        
        # 4. 消融实验结果
        if 'ablation_study' in self.results:
            ablation = self.results['ablation_study']
            
            labels = ['Original', 'Synonym\nReplacement']
            values = [ablation['original_accuracy'], ablation['synonym_accuracy']]
            
            axes[1, 1].bar(labels, values, color=['skyblue', 'lightcoral'])
            axes[1, 1].set_title('Ablation Study: Synonym Replacement')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_ylim([0, 1])
            
            for i, val in enumerate(values):
                axes[1, 1].text(i, val + 0.01, f'{val:.4f}', ha='center')
            
            # 添加准确率下降标注
            acc_drop = ablation['accuracy_drop']
            axes[1, 1].text(0.5, max(values) + 0.05, f'Accuracy Drop: {acc_drop:.4f}', 
                           ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.config['paths']['result_dir'], f'results_visualization_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {plot_path}")
        
        plt.show()
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.config['paths']['result_dir'], f'experiment_results_{timestamp}.json')
        
        # 准备要保存的结果（去除模型等大对象）
        saveable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                saveable_results[key] = {}
                for subkey, subvalue in value.items():
                    # 跳过模型和防御对象
                    if subkey not in ['model', 'tokenizer', 'fpp_defense']:
                        if isinstance(subvalue, (list, dict, int, float, str, bool)):
                            saveable_results[key][subkey] = subvalue
                        elif isinstance(subvalue, np.ndarray):
                            saveable_results[key][subkey] = subvalue.tolist()
                        elif torch.is_tensor(subvalue):
                            saveable_results[key][subkey] = subvalue.cpu().numpy().tolist()
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(saveable_results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {results_path}")
        
        # 同时保存为CSV格式的摘要
        summary_data = []
        
        # 基线结果
        for model_type in ['bert', 'bilstm']:
            key = f'baseline_{model_type}'
            if key in saveable_results:
                summary_data.append({
                    'Experiment': f'Baseline_{model_type.upper()}',
                    'Accuracy': saveable_results[key].get('test_accuracy', 0),
                    'Attack_Success_Rate': None,
                    'FPP_Accuracy': None,
                    'Rejection_Rate': None
                })
        
        # 攻击结果
        for key in saveable_results:
            if key.startswith('attack_'):
                attack_type = key.replace('attack_', '')
                metrics = saveable_results[key].get('attack_metrics', {})
                summary_data.append({
                    'Experiment': f'Attack_{attack_type.upper()}',
                    'Accuracy': metrics.get('adversarial_accuracy', 0),
                    'Attack_Success_Rate': metrics.get('attack_success_rate', 0),
                    'FPP_Accuracy': None,
                    'Rejection_Rate': None
                })
        
        # FPP防御结果
        if 'fpp_defense' in saveable_results:
            fpp_results = saveable_results['fpp_defense'].get('fpp_results', {})
            summary_data.append({
                'Experiment': 'FPP_Defense',
                'Accuracy': fpp_results.get('fpp_accuracy', 0),
                'Attack_Success_Rate': None,
                'FPP_Accuracy': fpp_results.get('fpp_accuracy', 0),
                'Rejection_Rate': fpp_results.get('rejection_rate', 0)
            })
        
        # 消融实验结果
        if 'ablation_study' in saveable_results:
            ablation = saveable_results['ablation_study']
            summary_data.append({
                'Experiment': 'Ablation_Study',
                'Accuracy': ablation.get('synonym_accuracy', 0),
                'Attack_Success_Rate': None,
                'FPP_Accuracy': None,
                'Rejection_Rate': None
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_path = os.path.join(self.config['paths']['result_dir'], f'results_summary_{timestamp}.csv')
            summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Results summary saved to {csv_path}")
            
            # 打印摘要
            print("\n" + "="*60)
            print("EXPERIMENT RESULTS SUMMARY")
            print("="*60)
            print(summary_df.to_string())