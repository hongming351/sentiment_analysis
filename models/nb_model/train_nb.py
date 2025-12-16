# train_nb.py - 朴素贝叶斯模型5折交叉验证训练
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import pickle
import time
import json
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

print("=" * 70)
print("京东评论情感分析 - 朴素贝叶斯模型5折交叉验证训练")
print("=" * 70)

# ==================== 文本预处理 ====================
def clean_text(text):
    """清洗文本"""
    if pd.isna(text):
        return ""
    
    import re
    text = str(text).strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：,.!?;\'"、]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_chinese(text, use_jieba=True):
    """中文分词"""
    import jieba
    text = clean_text(text)
    if not text:
        return ""
    
    if use_jieba:
        tokens = jieba.lcut(text)
    else:
        tokens = list(text)
    
    tokens = [token.strip() for token in tokens if token.strip()]
    return ' '.join(tokens)

# ==================== 加载数据 ====================
def load_fold_data(data_dir='data', n_folds=5):
    """加载交叉验证数据"""
    folds = []
    
    print(f"\n📂 加载数据详细信息:")
    print(f"  数据目录: {data_dir}")
    print(f"  目录是否存在: {os.path.exists(data_dir)}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 列出数据文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"  找到 {len(csv_files)} 个CSV文件")
    print(f"  文件列表: {csv_files[:10]}...")
    
    for fold_idx in range(n_folds):
        train_path = Path(data_dir) / f"train_fold_{fold_idx}.csv"
        val_path = Path(data_dir) / f"val_fold_{fold_idx}.csv"
        
        print(f"\n  处理第{fold_idx}折:")
        print(f"    训练文件: {train_path}")
        print(f"    是否存在: {train_path.exists()}")
        print(f"    验证文件: {val_path}")
        print(f"    是否存在: {val_path.exists()}")
        
        if not train_path.exists():
            raise FileNotFoundError(f"找不到第{fold_idx}折训练数据文件: {train_path}")
        
        if not val_path.exists():
            raise FileNotFoundError(f"找不到第{fold_idx}折验证数据文件: {val_path}")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        # 检查数据质量
        print(f"    训练集大小: {len(train_df)} 行")
        print(f"    验证集大小: {len(val_df)} 行")
        
        # 检查NaN值
        train_nan = train_df.isna().sum().sum()
        val_nan = val_df.isna().sum().sum()
        if train_nan > 0 or val_nan > 0:
            print(f"    警告: 训练集有 {train_nan} 个NaN，验证集有 {val_nan} 个NaN")
            # 清理NaN
            train_df = train_df.dropna(subset=['sentence', 'label'])
            val_df = val_df.dropna(subset=['sentence', 'label'])
            print(f"    清理后: 训练集 {len(train_df)} 行，验证集 {len(val_df)} 行")
        
        folds.append({
            'fold': fold_idx,
            'train': train_df,
            'val': val_df
        })
    
    # 加载测试集
    test_path = Path(data_dir) / "dev.csv"
    print(f"\n  测试集文件: {test_path}")
    print(f"  是否存在: {test_path.exists()}")
    
    if not test_path.exists():
        raise FileNotFoundError(f"找不到测试集文件: {test_path}")
    
    test_df = pd.read_csv(test_path)
    print(f"  测试集大小: {len(test_df)} 行")
    
    return folds, test_df

# ==================== 训练单折朴素贝叶斯 ====================
def train_nb_fold(fold_idx, train_df, val_df, config):
    """训练单个朴素贝叶斯模型"""
    print(f"\n📊 第 {fold_idx+1}/{config['n_folds']} 折训练")
    print("-" * 50)
    
    # 文本预处理
    print("文本预处理中...")
    train_texts = train_df[config['text_column']].apply(tokenize_chinese).tolist()
    train_labels = train_df[config['label_column']].astype(int).tolist()
    
    val_texts = val_df[config['text_column']].apply(tokenize_chinese).tolist()
    val_labels = val_df[config['label_column']].astype(int).tolist()
    
    # 创建朴素贝叶斯模型
    print("创建朴素贝叶斯模型中...")
    
    if config['vectorizer_type'] == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=(1, config['ngram_range']),
            min_df=config['min_df'],
            max_df=config['max_df'],
            sublinear_tf=True
        )
    else:  # count vectorizer
        vectorizer = CountVectorizer(
            max_features=config['max_features'],
            ngram_range=(1, config['ngram_range']),
            min_df=config['min_df'],
            max_df=config['max_df']
        )
    
    nb_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', MultinomialNB(
            alpha=config['alpha'],
            fit_prior=config['fit_prior']
        ))
    ])
    
    # 训练
    print("训练模型中...")
    start_time = time.time()
    nb_pipeline.fit(train_texts, train_labels)
    train_time = time.time() - start_time
    
    # 验证集评估
    print("验证集评估中...")
    val_predictions = nb_pipeline.predict(val_texts)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    
    # 保存模型
    model_dir = "nb_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"nb_fold_{fold_idx}.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(nb_pipeline, f)
    
    print(f"✓ 模型已保存到: {model_path}")
    print(f"  训练时间: {train_time:.2f}秒")
    print(f"  验证集准确率: {val_accuracy:.4f}")
    
    return {
        'fold': fold_idx,
        'model_path': model_path,
        'train_time': train_time,
        'val_accuracy': val_accuracy
    }

# ==================== 主函数 ====================
def main():
    import os
    
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 计算项目根目录
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # 构建正确的数据路径
    data_dir = os.path.join(project_root, 'data')
    
    # 配置参数
    config = {
        'data_dir': data_dir,
        'text_column': 'sentence',
        'label_column': 'label',
        'n_folds': 5,
        
        # 特征工程参数
        'vectorizer_type': 'tfidf',  # 'tfidf' 或 'count'
        'max_features': 10000,
        'ngram_range': 2,
        'min_df': 2,
        'max_df': 0.95,
        
        # 朴素贝叶斯参数
        'alpha': 1.0,  # 拉普拉斯平滑参数
        'fit_prior': True,  # 是否学习先验概率
        
        # 其他
        'seed': 42
    }
    
    print(f"\n📋 配置信息:")
    print(f"  当前脚本位置: {current_dir}")
    print(f"  项目根目录: {project_root}")
    print(f"  数据目录: {config['data_dir']}")
    print(f"  向量化器类型: {config['vectorizer_type']}")
    print(f"  最大特征数: {config['max_features']}")
    print(f"  n-gram范围: 1-{config['ngram_range']}")
    print(f"  平滑参数alpha: {config['alpha']}")
    
    # 加载数据
    print("\n" + "="*70)
    print("📊 加载数据")
    print("="*70)
    
    try:
        folds, test_df = load_fold_data(config['data_dir'], config['n_folds'])
        print(f"\n✓ 成功加载 {len(folds)} 折交叉验证数据")
        print(f"✓ 测试集: {len(test_df)} 条评论")
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("请检查数据文件路径是否正确。")
        return
    
    # 5折交叉验证训练
    print("\n" + "="*70)
    print("🚀 开始5折交叉验证训练")
    print("="*70)
    
    fold_results = []
    
    for fold_idx in range(config['n_folds']):
        fold_data = folds[fold_idx]
        result = train_nb_fold(
            fold_idx=fold_idx,
            train_df=fold_data['train'],
            val_df=fold_data['val'],
            config=config
        )
        fold_results.append(result)
    
    # 保存训练配置和结果
    results_dir = "nb_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(results_dir, 'nb_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # 计算统计信息
    val_accuracies = [r['val_accuracy'] for r in fold_results]
    
    # 保存训练结果摘要
    training_summary = {
        'fold_results': fold_results,
        'total_training_time': sum([r['train_time'] for r in fold_results]),
        'avg_training_time': np.mean([r['train_time'] for r in fold_results]),
        'val_accuracies': val_accuracies,
        'mean_val_accuracy': np.mean(val_accuracies),
        'std_val_accuracy': np.std(val_accuracies),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(results_dir, 'nb_training_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print("✅ 朴素贝叶斯训练完成总结")
    print("="*70)
    
    print(f"\n📊 训练统计:")
    print(f"  总训练时间: {training_summary['total_training_time']:.2f}秒")
    print(f"  平均每折训练时间: {training_summary['avg_training_time']:.2f}秒")
    print(f"  验证集准确率: {training_summary['mean_val_accuracy']:.4f} (±{training_summary['std_val_accuracy']:.4f})")
    print(f"  生成的模型: nb_models/nb_fold_0.pkl 到 nb_fold_4.pkl")
    
    print(f"\n📈 各折验证集准确率:")
    for i, result in enumerate(fold_results):
        print(f"  第{i+1}折: {result['val_accuracy']:.4f}")
    
    print(f"\n🚀 下一步:")
    print(f"  运行 evaluate_nb.py 评估模型性能")
    print(f"  运行 python evaluate_nb.py")

if __name__ == "__main__":
    main()