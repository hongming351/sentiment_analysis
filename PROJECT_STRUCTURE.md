# 京东评论情感分析系统 - 项目结构文档

## 当前项目结构

```
sentiment_analysis/
│
├── 📄 .gitignore                  # Git忽略配置
├── 📄 bert_sentiment_analyzer.py  # BERT情感分析主脚本
├── 📄 config.json                 # 项目配置文件
├── 📄 config.py                   # 配置管理类
├── 📄 PROJECT_STRUCTURE.md        # 项目结构文档（本文件）
├── 📄 README.md                   # 项目主文档
├── 📄 requirements.txt            # Python依赖包列表
├── 📄 run_visualization.py        # 可视化运行脚本
│
├── 📁 comprehensive_visualizations/  # 综合可视化报告
│   └── 📄 model_visualization_report.html  # 模型可视化HTML报告
│
├── 📁 data/                       # 数据处理模块
│   ├── 📄 data_loader.py          # 数据加载器
│   ├── 📄 explore_dataset.py      # 数据集探索工具
│   ├── 📄 jd.json                 # 数据集元数据配置
│   ├── 📄 k_fold.py               # 交叉验证工具
│   ├── 📄 model_source.py         # 模型源代码（LSTM实现）
│   ├── 📄 preprocess.py           # 文本预处理核心模块
│   ├── 📄 train.csv               # 训练数据集
│   └── 📄 dev.csv                 # 验证数据集
│
├── 📁 models/                     # 模型实现模块
│   ├── 📄 train_and_save_models.py  # 统一模型训练脚本
│   │
│   ├── 📁 bert_model/             # BERT模型实现
│   │   └── 📄 train_BERT.py       # BERT模型训练脚本
│   │
│   ├── 📁 lstm_model/             # LSTM+注意力模型
│   │   ├── 📄 config.json         # LSTM模型配置
│   │   └── 📄 train_lstm_global_vocab_visual.py  # LSTM训练脚本
│   │
│   ├── 📁 nb_model/               # 朴素贝叶斯模型
│   │   ├── 📄 evaluate_nb.py      # 朴素贝叶斯评估
│   │   └── 📄 train_nb.py         # 朴素贝叶斯训练
│   │
│   └── 📁 svm_model/              # SVM模型
│       ├── 📄 evaluate_svm.py     # SVM评估
│       └── 📄 train_svm.py        # SVM训练
│
└── 📁 utils/                      # 工具函数模块
    ├── 📄 attention.py            # 注意力机制核心实现
    └── 📄 visualization.py        # 可视化工具集
```

## 项目优化概述

### 已删除的文件和目录

1. **图片文件（34个）**：
   - 所有PNG可视化图片（性能曲线、混淆矩阵、训练历史等）
   - 来自comprehensive_visualizations、bert_visualizations和lstm_visualizations目录

2. **临时结果文件**：
   - 所有CSV报告文件（LSTM分类报告、模型统计等）
   - 所有JSON结果文件（训练摘要、模型比较等）
   - 临时脚本和报告文件

3. **空目录**：
   - competition_report/
   - nb_results/
   - bert_visualizations/
   - lstm_visualizations/

### 保留的核心组件

1. **核心代码文件**：
   - 所有Python实现文件（.py）
   - 配置文件和依赖管理
   - 数据处理和模型训练脚本

2. **必要数据文件**：
   - 数据加载和预处理模块
   - 元数据配置（jd.json）
   - 示例数据集（train.csv, dev.csv）

3. **关键报告文件**：
   - 保留了model_visualization_report.html作为综合可视化报告

## 当前项目统计

- **总文件数**：25个核心文件
- **总目录数**：8个组织良好的目录
- **数据记录**：50,398条评论数据（45,366训练 + 5,032验证）
- **模型类型**：4种算法（BERT, LSTM+Attention, 朴素贝叶斯, SVM）

## 使用说明

### 快速验证项目功能

```bash
# 测试数据加载
python -c "from data.data_loader import DataLoader; loader = DataLoader(); data = loader.load_jd_reviews(); print('数据加载成功:', len(data), '条记录')"

# 运行数据探索
python data/explore_dataset.py
```

### 模型训练示例

```bash
# 训练BERT模型
python models/bert_model/train_BERT.py

# 训练LSTM+Attention模型
python models/lstm_model/train_lstm_global_vocab_visual.py

# 训练朴素贝叶斯模型
python models/nb_model/train_nb.py

# 训练SVM模型
python models/svm_model/train_svm.py
```

## 目录功能说明

- **data/**：数据处理和加载功能
- **models/**：所有机器学习模型实现
- **utils/**：共享工具函数和辅助类
- **comprehensive_visualizations/**：保留的关键可视化报告

## 优化效果

1. **项目大小**：显著减小（删除30+图片文件和多个临时结果文件）
2. **结构清晰**：更加条理分明的目录组织
3. **核心功能**：所有关键功能保持完好
4. **易于维护**：更容易理解和管理的项目结构

---

**最后更新**：2026-01-08
**项目状态**：优化完成，核心功能验证通过