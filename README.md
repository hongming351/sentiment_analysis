# 京东评论情感分析系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于4种机器学习算法的京东评论情感分析系统，支持朴素贝叶斯、SVM、LSTM+注意力机制和BERT模型对比。

## 📊 模型性能

| 模型 | 准确率 | F1分数 | 训练时间 |
|------|--------|--------|----------|
| BERT | ~90% | ~90% | 约5小时 |
| LSTM+Attention | ~86% | ~86% | 约3小时 |
| 朴素贝叶斯 | ~86% | ~86% | ~5秒 |
| SVM | ~86% | ~86% | ~6秒 |

## 🚀 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
# SVM模型 (最快)
python models/svm_model/train_svm.py
python models/svm_model/evaluate_svm.py

# 朴素贝叶斯
python models/nb_model/train_nb.py
python models/nb_model/evaluate_nb.py

# LSTM (需要GPU)
python models/lstm_model/train_lstm_global_vocab_visual.py

# BERT (需要GPU，训练时间较长)
python models/bert_model/train_BERT.py
```

## 📁 项目结构

```
sentiment_analysis/
├── README.md                    # 项目文档
├── requirements.txt             # Python依赖
├── config.json                  # 配置文件
├── config.py                    # 配置管理
│
├── data/                        # 数据目录
│   ├── train.csv               # 训练数据
│   ├── dev.csv                 # 测试数据
│   ├── preprocess.py           # 数据预处理
│   ├── data_loader.py          # 数据加载
│   └── jd.json                 # 原始数据
│
├── models/                      # 模型目录
│   ├── svm_model/              # SVM模型
│   │   ├── train_svm.py        # 训练脚本
│   │   ├── evaluate_svm.py     # 评估脚本
│   │   ├── svm_models/         # 训练好的模型
│   │   └── results/            # 评估结果
│   │
│   ├── nb_model/               # 朴素贝叶斯
│   │   ├── train_nb.py         # 训练脚本
│   │   ├── evaluate_nb.py      # 评估脚本
│   │   ├── nb_models/          # 训练好的模型
│   │   └── nb_results/         # 评估结果
│   │
│   ├── lstm_model/             # LSTM模型
│   │   ├── train_lstm_global_vocab_visual.py
│   │   ├── config.json
│   │   ├── visualizations/     # 可视化结果
│   │   └── *.pt                # 训练好的模型
│   │
│   └── bert_model/             # BERT模型
│       ├── train_BERT.py       # 训练脚本
│       ├── bert_visualizations/ # 可视化结果
│       └── *.pth               # 训练好的模型
│
├── utils/                       # 工具模块
│   ├── attention.py            # 注意力机制
│   └── visualization.py        # 可视化工具
│
└── comprehensive_visualizations/ # 综合可视化报告
```

## 📈 使用说明

### 数据格式

CSV文件需包含 `sentence` 和 `label` 列：

```csv
sentence,label
"商品质量很好，满意！",1
"物流太慢了，差评",0
```

### 评估指标

- 准确率 (Accuracy)
- F1分数 (F1-Score)
- 精确率/召回率
- 混淆矩阵

## 📚 依赖包

- PyTorch, Transformers (深度学习)
- scikit-learn, numpy, pandas (机器学习)
- jieba (中文分词)
- matplotlib, seaborn (可视化)

## 📄 许可证

MIT License

**最后更新**: 2025-01-08
