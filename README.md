# 京东评论情感分析系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于4种机器学习算法的京东评论情感分析系统，支持朴素贝叶斯、SVM、LSTM+注意力机制和BERT模型对比。系统实现了完整的5折交叉验证训练流程，并提供详细的性能对比分析。

## 📊 最新性能结果

基于完整的5折交叉验证训练，系统在京东评论数据集上取得了以下性能：

| 排名 | 模型 | 准确率 | F1分数 | 训练时间 | 推理时间 | 参数量 | 特点 |
|------|------|--------|--------|----------|----------|---------|------|
| 🥇 1 | **BERT** | **90.70%** | **90.70%** | 17740s | 15.0ms | 19.1M | 预训练模型，语义理解最强 |
| 🥈 2 | **LSTM+Attention** | **86.15%** | **86.15%** | 10000s | 5.0ms | 2M | 自定义注意力机制，可解释性强 |
| 🥉 3 | **朴素贝叶斯** | **86.19%** | **86.18%** | 5.0s | 0.1ms | 2K | 训练最快，适合快速原型 |
| 4 | **SVM** | **85.49%** | **85.49%** | 6.1s | 0.5ms | 2K | 泛化能力强，决策边界清晰 |

### 性能亮点

- **BERT模型**表现最佳，在4.9小时训练后达到90.70%准确率，适合高精度要求的生产环境
- **LSTM+Attention**通过自定义注意力机制实现了86.15%准确率，模型可解释性突出
- **朴素贝叶斯**仅需5.0秒即可完成5折交叉验证训练，达到86.19%准确率，是快速原型开发的理想选择
- **SVM模型**在6.1秒训练后达到85.49%准确率，泛化能力强
- **集成学习**进一步提升了模型性能，SVM软投票集成可达86.01%准确率

## 🎯 主要特性

- **多算法对比**: 集成传统机器学习和深度学习算法
- **自定义注意力机制**: 为LSTM模型实现了创新的注意力机制
- **完整评估体系**: 提供准确率、F1分数、召回率等多维度评估
- **5折交叉验证**: 确保模型性能的可靠性和泛化能力
- **可视化分析**: 生成训练曲线、性能对比图表和雷达图
- **自动化报告**: 一键生成完整的比赛分析报告
- **模型集成**: 支持软投票和硬投票集成方法

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- 内存: 至少8GB RAM（用于深度学习模型）
- GPU: 可选，NVIDIA GPU（用于加速BERT训练）

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd jd_changed12.12

# 安装Python依赖
pip install -r requirements.txt

# 或者使用虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 运行完整分析

```bash
# 生成完整的比赛报告
python competition_report_fix_suggestions_complete.py

# 训练BERT模型（5折交叉验证）
python models/bert_model/train_BERT_fixed.py

# 训练LSTM+Attention模型
python models/lstm_model/train_lstm_global_vocab_visual.py

# 训练朴素贝叶斯模型
python models/nb_model/train_nb.py

# 训练SVM模型
python models/svm_model/train_svm.py
```

## 📁 项目架构

```
jd_changed12.12/
│
├── 📄 README.md                       # 项目文档
├── 📄 requirements.txt                # Python依赖包列表
├── 📄 config.json                     # 项目配置文件
├── 📄 config.py                       # 配置管理类
├── 📄 competition_report_fix_suggestions_complete.py # 主要报告生成程序
├── 📄 .gitignore                      # Git忽略文件
│
├── 📁 data/                           # 数据处理模块
│   ├── 📄 data_loader.py              # 数据加载器
│   ├── 📄 preprocess.py               # 文本预处理核心模块
│   ├── 📄 explore_dataset.py          # 数据探索工具
│   ├── 📄 model_source.py             # 模型源文件
│   ├── 📄 k_fold.py                   # 交叉验证工具
│   ├── 📄 dev.csv                     # 验证集
│   ├── 📄 train.csv                   # 训练集
│   ├── 📄 jd.json                     # 原始数据
│   └── 📁 train_fold_*.csv            # 5折训练数据
│
├── 📁 models/                         # 模型实现模块
│   ├── 📄 train_and_save_models.py    # 统一训练脚本
│   ├── 📁 bert_model/                 # BERT模型实现
│   │   ├── 📄 train_BERT_fixed.py     # BERT 5折交叉验证训练
│   │   └── 📄 train_BERT.py           # BERT基础训练脚本
│   ├── 📁 lstm_model/                 # LSTM+注意力模型
│   │   ├── 📄 train_lstm_global_vocab_visual.py # LSTM训练脚本
│   │   ├── 📄 jd_lstm_fold_*_best_global.pt     # 5折最佳模型
│   │   ├── 📄 vocab.pkl               # 词汇表
│   │   ├── 📄 lstm_results.json       # LSTM实验结果
│   │   └── 📁 visualizations/         # LSTM可视化结果
│   ├── 📁 nb_model/                   # 朴素贝叶斯模型
│   │   ├── 📄 train_nb.py             # 朴素贝叶斯训练
│   │   ├── 📄 evaluate_nb.py          # 朴素贝叶斯评估
│   │   ├── 📄 naive_bayes_model.pkl   # 朴素贝叶斯模型
│   │   ├── 📁 nb_models/              # 训练好的模型文件
│   │   └── 📁 nb_results/             # 朴素贝叶斯结果
│   └── 📁 svm_model/                  # SVM模型
│       ├── 📄 train_svm.py            # SVM训练
│       ├── 📄 evaluate_svm.py         # SVM评估
│       ├── 📄 svm_model.pkl           # SVM模型
│       ├── 📁 svm_models/             # 训练好的模型文件
│       └── 📁 svm_results/            # SVM结果
│
├── 📁 utils/                          # 工具函数模块
│   ├── 📄 attention.py                # 注意力机制核心实现
│   └── 📄 visualization.py            # 可视化工具集
│
├── 📁 bert_fold_*_best_transformers/  # BERT Transformers格式模型（5折）
├── 📁 bert_visualizations/            # BERT训练可视化结果
└── 📁 competition_report/             # 比赛报告输出目录
    ├── 📄 competition_report.txt      # 文本格式报告
    ├── 📄 fix_report.md               # 修复报告
    ├── 📄 model_comparison.csv        # 模型对比表格
    ├── 📄 model_results.json          # 性能数据
    ├── 📄 run_complete.py             # 完整分析运行脚本
    ├── 📄 updated_model_comparison.py # 更新版模型对比
    └── 📁 result_comparison/          # 详细对比结果
        ├── 📄 bert_vs_lstm_training_comparison.png
        ├── 📄 model_comparison_report.json
        ├── 📄 detailed_model_comparison.csv
        ├── 📄 detailed_model_comparison.html
        ├── 📄 model_accuracy_comparison.png
        └── 📄 model_radar_chart.png
```

## 🔬 技术创新

### 1. 自定义注意力机制

```python
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        # 计算注意力权重
        attention_weights = torch.softmax(
            self.attention(lstm_output), dim=1
        )
        # 加权求和得到上下文向量
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights
```

### 2. 多层次文本预处理

- 中文分词（使用jieba）
- 停用词过滤
- 文本标准化处理
- 词汇表构建

### 3. 统一评估框架

- 交叉验证评估
- 混淆矩阵分析
- 学习曲线绘制
- 性能指标计算

### 4. 集成学习策略

- **软投票**: 平均各模型预测概率，提升泛化能力
- **硬投票**: 基于多数票决策，提高稳定性

## 📈 详细性能分析

### 训练效率对比

```
朴素贝叶斯: ██ 5.0秒        ████████████████████████████ 86.19%
SVM:       ████ 6.1s        ██████████████████████████ 85.49%
LSTM:      ████████████████ 10000s      ████████████████████████ 86.15%
BERT:      █████████████████████████████ 17740s       ████████████████████████████████ 90.70%
```

### 模型推荐场景

| 应用场景 | 推荐模型 | 理由 | 性能预期 |
|----------|----------|------|----------|
| **高精度生产环境** | BERT | 准确率最高（90.70%），语义理解强 | 90%+ |
| **快速原型开发** | 朴素贝叶斯 | 训练最快（5.0秒），部署简单 | 86%+ |
| **需要模型解释** | LSTM+Attention | 注意力可视化，可理解决策过程 | 86%+ |
| **平衡各方面需求** | SVM | 训练时间适中，泛化能力强 | 85%+ |

### 关键发现

1. **BERT模型性能最佳**，在4.9小时训练后达到90.70%准确率
2. **LSTM+Attention模型**在10个epoch后达到86.15%准确率，模型可解释性强
3. **朴素贝叶斯训练最快**，仅需5.0秒完成5折交叉验证，达到86.19%准确率，适合快速原型开发
4. **SVM模型**在6.1秒训练后达到85.49%准确率，泛化能力强
5. **集成学习**可进一步提升性能，SVM软投票集成可达86.01%准确率
6. **注意力机制可视化**帮助理解模型决策过程

## 📚 依赖包说明

### 核心依赖

- **深度学习**: PyTorch, Transformers
- **机器学习**: scikit-learn, numpy, pandas
- **中文处理**: jieba, NLTK
- **可视化**: matplotlib, seaborn
- **进度显示**: tqdm
- **安全存储**: safetensors（用于BERT模型保存）

详细版本信息请参考`requirements.txt`文件。

## 🛠️ 使用说明

### 数据准备

项目支持以下数据格式：

- CSV文件：包含`sentence`和`label`列
- JSON文件：结构化文本数据

```python
# 示例数据结构
sentence,label
"这个商品质量很好，值得购买！",1
"服务态度差，不推荐",0
```

### 自定义配置

编辑`config.json`文件来自定义训练参数：

```json
{
  "training": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "models": {
    "LSTM": {
      "embedding_dim": 256,
      "hidden_dim": 128,
      "use_attention": true
    }
  }
}
```

### 生成报告

```python
from competition_report_fix_suggestions_complete import CompetitionReport

# 创建报告生成器
reporter = CompetitionReport()

# 生成完整分析报告
reporter.generate_report()
```

## 📁 输出文件

运行完成后，将在`competition_report/`目录生成：

- `final_analysis_report.md` - Markdown格式分析报告
- `model_comparison.csv` - 模型对比表格
- `model_results.json` - 性能数据（JSON格式）
- `model_comparison_visualization.png` - 综合性能图表
- `detailed_model_comparison.html` - 交互式HTML报告

## 📄 许可证

本项目采用MIT许可证。

## 🏆 致谢

感谢以下开源项目：

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Transformers](https://huggingface.co/transformers/) - 预训练模型库
- [jieba](https://github.com/fxsjy/jieba) - 中文分词
- [scikit-learn](https://scikit-learn.org/) - 机器学习库

---

**最后更新**: 2025-12-15  
**项目版本**: v2.1  
**数据集**: 京东评论情感分析数据集  
**训练方式**: 5折交叉验证
