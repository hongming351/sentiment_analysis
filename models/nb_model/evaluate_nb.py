# evaluate_nb.py - 朴素贝叶斯模型评估
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("京东评论情感分析 - 朴素贝叶斯模型评估")
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

# ==================== 加载测试数据 ====================
def load_test_data(data_dir=None):
    """加载测试数据"""
    import os
    import pandas as pd
    
    if data_dir is None:
        # 自动计算绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(project_root, 'data')
    
    test_path = os.path.join(data_dir, "dev.csv")
    print(f"测试文件路径: {test_path}")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试集文件: {test_path}")
    
    # 加载数据
    test_df = pd.read_csv(test_path)
    print(f"原始测试集: {len(test_df)} 条")
    
    # 检查并清理NaN
    original_len = len(test_df)
    test_df_clean = test_df.dropna(subset=['sentence', 'label'])
    
    print(f"清理后测试集: {len(test_df_clean)} 条")
    print(f"移除了 {original_len - len(test_df_clean)} 条包含NaN的数据")
    
    if len(test_df_clean) == 0:
        raise ValueError("清理后没有有效数据")
    
    # 检查标签分布
    print("\n📊 标签分布统计:")
    label_counts = test_df_clean['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = count / len(test_df_clean) * 100
        print(f"  标签 {label}: {count} 条 ({percentage:.1f}%)")
    
    return test_df_clean

# ==================== 加载模型 ====================
def load_nb_models(model_dir=None, n_folds=5):
    """加载所有朴素贝叶斯模型"""
    import os
    
    if model_dir is None:
        # 自动计算绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # 按优先级尝试多个路径
        possible_dirs = [
            os.path.join(project_root, 'nb_models'),  # 根目录的nb_models
            os.path.join(current_dir, 'nb_models'),   # 当前目录的nb_models
            os.path.join(project_root, 'models', 'nb_model', 'nb_models'),
        ]
        
        model_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and any(f'nb_fold_{i}.pkl' in os.listdir(dir_path) for i in range(n_folds)):
                model_dir = dir_path
                break
        
        if model_dir is None:
            model_dir = possible_dirs[0]  # 默认使用第一个
    
    print(f"\n📂 模型目录: {model_dir}")
    print(f"   目录是否存在: {os.path.exists(model_dir)}")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    
    # 列出目录内容
    try:
        files = os.listdir(model_dir)
        print(f"   目录内容 ({len(files)} 个文件):")
        for file in files:
            if file.endswith('.pkl'):
                print(f"     - {file}")
    except Exception as e:
        print(f"   无法列出目录内容: {e}")
    
    models = []
    loaded_count = 0
    
    for fold_idx in range(n_folds):
        model_path = os.path.join(model_dir, f"nb_fold_{fold_idx}.pkl")
        
        if not os.path.exists(model_path):
            print(f"⚠️  警告: 找不到模型文件 {model_path}")
            continue
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                models.append(model)
            print(f"✓ 加载第{fold_idx+1}折模型成功")
            loaded_count += 1
        except Exception as e:
            print(f"❌ 加载第{fold_idx+1}折模型失败: {e}")
    
    if loaded_count == 0:
        raise FileNotFoundError("未能加载任何模型")
    
    print(f"\n✅ 总共加载了 {loaded_count} 个模型")
    return models

# ==================== 评估模型 ====================
def evaluate_nb_models(nb_models, test_texts, test_labels):
    """评估朴素贝叶斯模型性能"""
    import numpy as np
    
    print(f"\n🔍 数据检查:")
    print(f"  测试文本数量: {len(test_texts)}")
    print(f"  测试标签数量: {len(test_labels)}")
    
    # 确保标签是整数类型
    test_labels = [int(label) for label in test_labels]
    
    # 转换为numpy数组
    test_labels_np = np.array(test_labels)
    
    print(f"\n📊 标签分布:")
    unique_labels, counts = np.unique(test_labels_np, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  标签 {label}: {count} 条 ({count/len(test_labels_np)*100:.1f}%)")
    
    # 收集所有预测
    all_predictions = []
    
    print(f"\n🤖 开始预测 ({len(nb_models)} 个模型)...")
    for i, model in enumerate(nb_models):
        print(f"  模型 {i+1}/{len(nb_models)} 预测中...")
        predictions = model.predict(test_texts)
        all_predictions.append(predictions)
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    
    # 投票集成
    print("  进行投票集成...")
    ensemble_predictions = []
    
    for i in range(len(test_texts)):
        preds = all_predictions[:, i]
        # 取众数
        unique, counts = np.unique(preds, return_counts=True)
        ensemble_predictions.append(unique[np.argmax(counts)])
    
    ensemble_predictions = np.array(ensemble_predictions)
    
    # 使用第一个模型的单个预测
    single_predictions = all_predictions[0]
    
    # 计算指标
    print("\n📈 计算评估指标...")
    
    # 集成模型指标
    ensemble_acc = accuracy_score(test_labels_np, ensemble_predictions)
    ensemble_precision = precision_score(test_labels_np, ensemble_predictions, average='weighted', zero_division=0)
    ensemble_recall = recall_score(test_labels_np, ensemble_predictions, average='weighted', zero_division=0)
    ensemble_f1 = f1_score(test_labels_np, ensemble_predictions, average='weighted', zero_division=0)
    
    # 单个模型指标
    single_acc = accuracy_score(test_labels_np, single_predictions)
    single_precision = precision_score(test_labels_np, single_predictions, average='weighted', zero_division=0)
    single_recall = recall_score(test_labels_np, single_predictions, average='weighted', zero_division=0)
    single_f1 = f1_score(test_labels_np, single_predictions, average='weighted', zero_division=0)
    
    # 分类报告
    print("\n📋 集成模型分类报告:")
    print(classification_report(test_labels_np, ensemble_predictions, digits=4))
    
    print("📋 单个模型分类报告:")
    print(classification_report(test_labels_np, single_predictions, digits=4))
    
    # 混淆矩阵
    ensemble_cm = confusion_matrix(test_labels_np, ensemble_predictions)
    single_cm = confusion_matrix(test_labels_np, single_predictions)
    
    return {
        'ensemble': {
            'accuracy': ensemble_acc,
            'precision': ensemble_precision,
            'recall': ensemble_recall,
            'f1': ensemble_f1,
            'predictions': ensemble_predictions,
            'confusion_matrix': ensemble_cm
        },
        'single': {
            'accuracy': single_acc,
            'precision': single_precision,
            'recall': single_recall,
            'f1': single_f1,
            'predictions': single_predictions,
            'confusion_matrix': single_cm
        },
        'test_labels': test_labels_np,
        'test_size': len(test_texts)
    }

# ==================== 可视化结果 ====================
def plot_results(results, save_dir='nb_evaluation_results'):
    """可视化评估结果"""
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("⚠️  警告: 中文字体设置失败，图表可能无法显示中文")
    
    # 1. 性能指标对比图
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('朴素贝叶斯模型性能对比', fontsize=16)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['准确率', '精确率', '召回率', 'F1分数']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row = i // 2
        col = i % 2
        
        ensemble_value = results['ensemble'][metric]
        single_value = results['single'][metric]
        
        ax[row, col].bar(['集成模型', '单个模型'], [ensemble_value, single_value], 
                        color=['skyblue', 'lightcoral'])
        ax[row, col].set_title(title)
        ax[row, col].set_ylim([0, 1])
        
        # 在柱子上添加数值
        for j, value in enumerate([ensemble_value, single_value]):
            ax[row, col].text(j, value + 0.02, f'{value:.4f}', 
                            ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 性能对比图已保存: {save_dir}/performance_comparison.png")
    
    # 2. 混淆矩阵
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('混淆矩阵对比', fontsize=16)
    
    # 集成模型混淆矩阵
    sns.heatmap(results['ensemble']['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('集成模型')
    ax1.set_xlabel('预测标签')
    ax1.set_ylabel('真实标签')
    
    # 单个模型混淆矩阵
    sns.heatmap(results['single']['confusion_matrix'], 
                annot=True, fmt='d', cmap='Reds', ax=ax2)
    ax2.set_title('单个模型')
    ax2.set_xlabel('预测标签')
    ax2.set_ylabel('真实标签')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵图已保存: {save_dir}/confusion_matrices.png")
    
    # 3. 错误分析图
    errors = results['ensemble']['predictions'] != results['test_labels']
    error_indices = np.where(errors)[0]
    
    if len(error_indices) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        error_labels_pred = results['ensemble']['predictions'][error_indices]
        error_labels_true = results['test_labels'][error_indices]
        
        error_pairs = list(zip(error_labels_true, error_labels_pred))
        unique_pairs, pair_counts = np.unique(error_pairs, axis=0, return_counts=True)
        
        # 创建标签
        pair_labels = [f'{true}→{pred}' for true, pred in unique_pairs]
        
        ax.bar(range(len(unique_pairs)), pair_counts, color='salmon')
        ax.set_title('错误类型分析 (真实标签→预测标签)')
        ax.set_xlabel('错误类型')
        ax.set_ylabel('错误数量')
        ax.set_xticks(range(len(unique_pairs)))
        ax.set_xticklabels(pair_labels, rotation=45, ha='right')
        
        # 添加数量标签
        for i, count in enumerate(pair_counts):
            ax.text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"✓ 错误分析图已保存: {save_dir}/error_analysis.png")
    
    plt.close('all')
    print(f"\n✅ 所有图表已保存到: {save_dir}")

# ==================== 主函数 ====================
# ==================== 主函数 ====================
def main():
    import os
    
    # 配置参数
    config = {
        'data_dir': None,  # 自动检测
        'model_dir': None,  # 自动检测
        'text_column': 'sentence',
        'label_column': 'label',
        'n_folds': 5,
        'results_dir': 'nb_evaluation_results',
        'seed': 42
    }
    
    print(f"\n📋 配置信息:")
    for key, value in config.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    # 创建结果目录
    os.makedirs(config['results_dir'], exist_ok=True)
    print(f"✓ 创建结果目录: {config['results_dir']}")
    
    # 加载测试数据
    print("\n" + "="*70)
    print("📊 加载测试数据")
    print("="*70)
    
    try:
        test_df = load_test_data(config['data_dir'])
        print(f"✓ 测试集加载完成: {len(test_df)} 条数据")
    except Exception as e:
        print(f"❌ 错误: {e}")
        return
    
    # 文本预处理
    print("\n🔄 文本预处理中...")
    test_texts = test_df[config['text_column']].apply(tokenize_chinese).tolist()
    test_labels = test_df[config['label_column']].tolist()
    
    print(f"  预处理后文本数量: {len(test_texts)}")
    print(f"  预处理后标签数量: {len(test_labels)}")
    
    # 加载模型
    print("\n" + "="*70)
    print("🤖 加载朴素贝叶斯模型")
    print("="*70)
    
    try:
        nb_models = load_nb_models(config['model_dir'], config['n_folds'])
        print(f"✓ 加载了 {len(nb_models)} 个朴素贝叶斯模型")
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请先运行 train_nb.py 训练模型")
        return
    
    # 评估模型
    print("\n" + "="*70)
    print("📈 评估模型性能")
    print("="*70)
    
    results = evaluate_nb_models(nb_models, test_texts, test_labels)
    
    # 显示结果
    print("\n" + "="*70)
    print("📊 评估结果总结")
    print("="*70)
    
    print(f"\n🎯 集成模型 (投票法):")
    print(f"  准确率: {results['ensemble']['accuracy']:.4f}")
    print(f"  精确率: {results['ensemble']['precision']:.4f}")
    print(f"  召回率: {results['ensemble']['recall']:.4f}")
    print(f"  F1分数: {results['ensemble']['f1']:.4f}")
    
    print(f"\n🎯 单个模型 (第1折):")
    print(f"  准确率: {results['single']['accuracy']:.4f}")
    print(f"  精确率: {results['single']['precision']:.4f}")
    print(f"  召回率: {results['single']['recall']:.4f}")
    print(f"  F1分数: {results['single']['f1']:.4f}")
    
    print(f"\n📈 性能提升:")
    accuracy_improvement = results['ensemble']['accuracy'] - results['single']['accuracy']
    f1_improvement = results['ensemble']['f1'] - results['single']['f1']
    print(f"  准确率提升: {accuracy_improvement:.4f}")
    print(f"  F1分数提升: {f1_improvement:.4f}")
    
    # 保存结果
    print("\n💾 保存评估结果...")
    
    # 转换函数：将numpy对象转换为Python原生类型
    def convert_for_json(obj):
        """递归转换对象为JSON可序列化的格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    # 准备保存的数据
    results_to_save = {
        'config': config,
        'metrics': {
            'ensemble': {
                'accuracy': float(results['ensemble']['accuracy']),
                'precision': float(results['ensemble']['precision']),
                'recall': float(results['ensemble']['recall']),
                'f1': float(results['ensemble']['f1']),
                'confusion_matrix': convert_for_json(results['ensemble']['confusion_matrix'])
            },
            'single': {
                'accuracy': float(results['single']['accuracy']),
                'precision': float(results['single']['precision']),
                'recall': float(results['single']['recall']),
                'f1': float(results['single']['f1']),
                'confusion_matrix': convert_for_json(results['single']['confusion_matrix'])
            }
        },
        'test_info': {
            'size': results['test_size'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'performance_improvement': {
            'accuracy_improvement': float(accuracy_improvement),
            'f1_improvement': float(f1_improvement)
        }
    }
    
    results_file = os.path.join(config['results_dir'], 'nb_evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 评估结果已保存: {results_file}")
    
    # 可视化
    print("\n🎨 生成可视化图表...")
    try:
        plot_results(results, save_dir=config['results_dir'])
    except Exception as e:
        print(f"⚠️  可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ 朴素贝叶斯模型评估完成")
    print("="*70)
    
    print(f"\n📁 生成的文件:")
    print(f"  评估结果: {config['results_dir']}/nb_evaluation_results.json")
    print(f"  性能对比图: {config['results_dir']}/performance_comparison.png")
    print(f"  混淆矩阵图: {config['results_dir']}/confusion_matrices.png")
    print(f"  错误分析图: {config['results_dir']}/error_analysis.png")
    
    print(f"\n🚀 下一步建议:")
    print("  1. 查看生成的图表分析模型性能")
    print("  2. 对比SVM和朴素贝叶斯模型的结果")
    print("  3. 根据错误分析优化模型或数据")

if __name__ == "__main__":
    main()
