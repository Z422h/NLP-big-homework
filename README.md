# 基于FPP框架的欺诈对话对抗数据生成与防御实验

## 项目简介

本项目基于论文《词级稳健性增强：以扰动对抗扰动》提出的FPP框架，实现了欺诈对话检测中的对抗数据生成与防御系统。项目包含完整的实验流程：从数据预处理、模型训练、对抗攻击生成，到FPP防御实现和实验结果分析。

## 项目目标

评估BERT、BiLSTM等模型在欺诈对话检测任务上的基线性能
分析对抗性数据改写对模型性能的影响
验证FPP防御框架在欺诈检测场景的有效性
对比不同攻击策略的效果差异

## 环境配置

### 系统要求
Python 3.9+
CUDA 11.8 (GPU训练推荐)
至少16GB RAM

### 安装依赖
```bash
pip install -r requirements.txt
```
### 主要依赖包
torch==2.0.1
transformers==4.30.0
numpy==1.24.3
pandas==1.5.3
scikit-learn==1.2.2
jieba==0.42.1
matplotlib==3.7.1
textattack==0.3.8

## 数据集说明

### 数据来源
使用课堂提供的欺诈对话检测数据集，包含：
训练集：14,363条对话样本
测试集：2,677条对话样本

### 数据特征
字段：specific_dialogue_content (对话内容), is_fraud (欺诈标签)
类别分布：欺诈样本约占52%，基本平衡

### 数据预处理
文本清洗：去除特殊字符，保留中文、英文和数字
分词处理：使用jieba进行中文分词
标签转换：将布尔/字符串标签转换为0/1数值标签

## 快速开始

### 运行完整实验
```bash
python complete_final_experiment.py
```
### 运行强力攻击实验
```bash
python run_experiment.py
```

### 实验流程
数据加载与预处理：加载CSV数据，进行清洗和转换
模型训练：训练BERT和BiLSTM基线模型
对抗攻击测试：使用多种策略生成对抗样本
FPP防御评估：测试FPP防御框架效果
结果可视化：生成图表和分析报告

### 项目结构
```bash
NLP-big-homework/
├── configs/                    # 配置文件目录
│   ├── config.yaml            # 主配置文件
│   └── config_fixed.yaml      # 修复后的配置文件
├── data/                      # 数据目录
│   ├── 测试集结果.csv         # 测试数据集
│   └── 训练集结果.csv         # 训练数据集
├── models/                    # 模型文件目录
│   └── bert-base-chinese/     # BERT预训练模型
├── results/                   # 实验结果目录
│   ├── comprehensive_final_results.json      # 完整实验结果
│   └── comprehensive_final_visualization.png # 实验结果可视化
├── strong_attack_results/     # 强力攻击实验结果
│   ├── *.png                  # 可视化图表
│   ├── *.csv                  # 详细结果数据
│   └── strong_attack_report.txt # 强力攻击实验报告
├── src/                       # 源代码目录
│   ├── attack_methods.py      # 对抗攻击方法
│   ├── data_preprocessing.py  # 数据预处理
│   ├── experiment.py          # 实验运行器
│   ├── fpp_defense.py         # FPP防御实现
│   ├── models.py              # 模型定义
│   └── utils.py               # 工具函数
├── complete_final_experiment.py # 完整实验脚本
├── run_experiment.py          # 强力攻击实验脚本
├── requirements.txt           # 依赖包列表
└── README.md                  # 项目说明文档
```

###  实验设计

1. 基线实验
BERT模型：基于bert-base-chinese进行微调
BiLSTM模型：传统序列模型对比
传统机器学习模型：LogisticRegression、RandomForest、SVM
2. 对抗攻击实验
攻击策略
同义词替换：替换关键欺诈词汇
句子替换：重写关键句子结构
插入删除攻击：添加/删除文本内容
综合攻击：组合多种攻击方式
评估指标:
原始准确率
攻击后准确率
准确率下降幅度
攻击成功率
3. FPP防御实验
防御框架：基于FPP（Fight Perturbation with Perturbation）
测试条件：针对insert_delete攻击策略
评估指标：防御准确率、改进幅度、计算开销

### 结果展示
complete_final_experiment.py的结果保存在result文件夹
run_experiment.py的结果保存在strong_attack_results文件夹