# evaluate_svm.py
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import pickle
import time
import json
import matplotlib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STXihei']
matplotlib.rcParams['axes.unicode_minus'] = False
print("=" * 70)
print("京东评论情感分析 - SVM模型评估")
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
def load_test_data(data_dir=None):
    """加载测试数据"""
    import os
    import pandas as pd
    
    if data_dir is None:
        data_dir = r"D:\jd_changed12.11\data"
    
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
def load_svm_models(n_folds=5):
    """加载所有SVM模型 - 智能路径检测"""
    import os
    
    # 可能的模型路径列表（按优先级排序）
    possible_dirs = [
        # 1. 硬编码绝对路径
        r"D:\jd_changed12.11\models\svm_model\svm_models",
        
        # 2. 相对于当前脚本的路径
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "svm_models"),
        
        # 3. 相对于当前工作目录的路径
        os.path.join(os.getcwd(), "svm_models"),
        os.path.join(os.getcwd(), "models", "svm_model", "svm_models"),
        
        # 4. 其他可能的路径
        "svm_models",
        "../svm_models",
        "../../svm_models",
    ]
    
    # 查找存在的目录
    model_dir = None
    for dir_path in possible_dirs:
        print(f"检查目录: {dir_path}")
        if os.path.exists(dir_path):
            # 检查目录中是否有模型文件
            has_models = any(f"svm_fold_{i}.pkl" in os.listdir(dir_path) 
                           for i in range(n_folds) if os.path.exists(dir_path))
            if has_models:
                model_dir = dir_path
                print(f"✓ 找到模型目录: {model_dir}")
                break
    
    if model_dir is None:
        raise FileNotFoundError("找不到包含SVM模型的目录")
    
    print(f"使用模型目录: {model_dir}")
    print(f"目录内容: {os.listdir(model_dir)}")
    
    models = []
    
    for fold_idx in range(n_folds):
        model_path = os.path.join(model_dir, f"svm_fold_{fold_idx}.pkl")
        
        if not os.path.exists(model_path):
            print(f"⚠️ 警告: 找不到模型文件 {model_path}")
            # 跳过这个模型，继续加载其他的
            continue
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                models.append(model)
            print(f"✓ 加载第{fold_idx+1}折模型成功")
        except Exception as e:
            print(f"❌ 加载第{fold_idx+1}折模型失败: {e}")
    
    if len(models) == 0:
        raise FileNotFoundError("未能加载任何模型")
    
    print(f"成功加载了 {len(models)} 个模型")
    return models
# ==================== 集成预测 ====================
def ensemble_predict(models, texts, voting='soft'):
    """集成预测"""
    if voting == 'hard':
        # 硬投票
        all_predictions = []
        for model in models:
            pred = model.predict(texts)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        final_predictions = []
        
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            vote_counts = np.bincount(votes)
            final_predictions.append(np.argmax(vote_counts))
        
        return np.array(final_predictions)
    
    else:
        # 软投票
        try:
            all_probs = []
            for model in models:
                probs = model.decision_function(texts)
                # 将决策函数值转换为概率
                probs = 1 / (1 + np.exp(-probs))
                probs = np.column_stack([1-probs, probs])  # 转换为二分类概率
                all_probs.append(probs)
            
            avg_probs = np.mean(all_probs, axis=0)
            return np.argmax(avg_probs, axis=1)
        except:
            print("⚠️  无法进行软投票，使用硬投票")
            return ensemble_predict(models, texts, voting='hard')

# ==================== 可视化函数 ====================
def plot_confusion_matrix(cm, title='混淆矩阵', save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'],
                yticklabels=['负面', '正面'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {save_path}")
    
    plt.show()

def plot_performance_comparison(single_acc, ensemble_soft_acc, ensemble_hard_acc, save_path=None):
    """绘制性能对比图"""
    models = ['单模型', '集成模型\n(软投票)', '集成模型\n(硬投票)']
    accuracies = [single_acc * 100, ensemble_soft_acc * 100, ensemble_hard_acc * 100]
    improvements = [0, (ensemble_soft_acc - single_acc) * 100, (ensemble_hard_acc - single_acc) * 100]
    
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for i, (bar, acc, imp) in enumerate(zip(bars, accuracies, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
               f'{acc:.2f}%', ha='center', va='bottom', fontsize=11)
        
        if i > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'+{imp:.2f}%', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='red')
    
    ax.set_ylabel('测试准确率 (%)', fontsize=12)
    ax.set_title('SVM模型性能对比', fontsize=14, fontweight='bold')
    ax.set_ylim([min(accuracies)-2, max(accuracies)+2])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 性能对比图已保存: {save_path}")
    
    plt.show()

# ==================== 主函数 ====================
def main():
    # 配置
    config = {
        'data_dir': r"D:\jd_changed12.11\data",
        'model_dir': r"D:\jd_changed12.11\models\svm_model\svm_models",
        'text_column': 'sentence',
        'label_column': 'label',
        'n_folds': 5,
        'seed': 42,
        'results_dir': 'evaluation_results' 
    }
    
    # 创建结果目录
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 加载测试数据
    print("\n" + "="*70)
    print("📊 加载测试数据")
    print("="*70)
    
    try:
        test_df = load_test_data(config['data_dir'])
        print(f"✓ 测试集: {len(test_df)} 条评论")
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return
    
    # 文本预处理
    print("文本预处理中...")
    test_texts = test_df[config['text_column']].apply(tokenize_chinese).tolist()
    test_labels = test_df[config['label_column']].tolist()
    
    print(f"✓ 预处理完成，有效样本: {len(test_texts)}")
    
    # 加载模型
    print("\n" + "="*70)
    print("🤖 加载SVM模型")
    print("="*70)
    
    try:
        svm_models = load_svm_models(n_folds=config['n_folds'])
        print(f"✓ 成功加载 {len(svm_models)} 个模型")
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print(f"请先运行 train_svm.py 训练模型")
        return
    
    # 评估单模型（第一折）
    print("\n" + "="*70)
    print("📈 评估单模型（第一折）")
    print("="*70)
    
    single_model = svm_models[0]
    single_predictions = single_model.predict(test_texts)
    single_acc = accuracy_score(test_labels, single_predictions)
    single_f1 = f1_score(test_labels, single_predictions, average='weighted')
    
    print(f"单模型测试准确率: {single_acc*100:.2f}%")
    print(f"单模型测试F1-score: {single_f1*100:.2f}%")
    
    # 计算单模型混淆矩阵
    single_cm = confusion_matrix(test_labels, single_predictions)
    
    # 评估集成模型（软投票）
    print("\n" + "="*70)
    print("🤝 评估集成模型（软投票）")
    print("="*70)
    
    ensemble_soft_predictions = ensemble_predict(svm_models, test_texts, voting='soft')
    ensemble_soft_acc = accuracy_score(test_labels, ensemble_soft_predictions)
    ensemble_soft_f1 = f1_score(test_labels, ensemble_soft_predictions, average='weighted')
    
    print(f"集成模型（软投票）测试准确率: {ensemble_soft_acc*100:.2f}%")
    print(f"集成模型（软投票）测试F1-score: {ensemble_soft_f1*100:.2f}%")
    print(f"相较于单模型提升: +{(ensemble_soft_acc - single_acc)*100:.2f}%")
    
    # 计算集成模型混淆矩阵
    ensemble_soft_cm = confusion_matrix(test_labels, ensemble_soft_predictions)
    
    # 评估集成模型（硬投票）
    print("\n" + "="*70)
    print("🤝 评估集成模型（硬投票）")
    print("="*70)
    
    ensemble_hard_predictions = ensemble_predict(svm_models, test_texts, voting='hard')
    ensemble_hard_acc = accuracy_score(test_labels, ensemble_hard_predictions)
    ensemble_hard_f1 = f1_score(test_labels, ensemble_hard_predictions, average='weighted')
    
    print(f"集成模型（硬投票）测试准确率: {ensemble_hard_acc*100:.2f}%")
    print(f"集成模型（硬投票）测试F1-score: {ensemble_hard_f1*100:.2f}%")
    print(f"相较于单模型提升: +{(ensemble_hard_acc - single_acc)*100:.2f}%")
    
    # 计算集成模型混淆矩阵
    ensemble_hard_cm = confusion_matrix(test_labels, ensemble_hard_predictions)
    
    # 生成分类报告
    print("\n" + "="*70)
    print("📋 详细分类报告（集成模型-软投票）")
    print("="*70)
    
    print(classification_report(test_labels, ensemble_soft_predictions,
                               target_names=['负面', '正面'],
                               digits=4))
    
    # ==================== 可视化 ====================
    print("\n" + "="*70)
    print("📊 生成可视化图表")
    print("="*70)
    
    # 1. 混淆矩阵
    plot_confusion_matrix(single_cm, 
                         title='SVM单模型混淆矩阵',
                         save_path=os.path.join(config['results_dir'], 'svm_single_cm.png'))
    
    plot_confusion_matrix(ensemble_soft_cm,
                         title='SVM集成模型（软投票）混淆矩阵',
                         save_path=os.path.join(config['results_dir'], 'svm_ensemble_soft_cm.png'))
    
    plot_confusion_matrix(ensemble_hard_cm,
                         title='SVM集成模型（硬投票）混淆矩阵',
                         save_path=os.path.join(config['results_dir'], 'svm_ensemble_hard_cm.png'))
    
    # 2. 性能对比图
    plot_performance_comparison(single_acc, ensemble_soft_acc, ensemble_hard_acc,
                               save_path=os.path.join(config['results_dir'], 'svm_performance_comparison.png'))
    
    # ==================== 保存结果 ====================
    print("\n" + "="*70)
    print("💾 保存评估结果")
    print("="*70)
    
    evaluation_results = {
        'single_model': {
            'accuracy': float(single_acc),
            'f1_score': float(single_f1),
            'confusion_matrix': single_cm.tolist()
        },
        'ensemble_soft': {
            'accuracy': float(ensemble_soft_acc),
            'f1_score': float(ensemble_soft_f1),
            'improvement': float(ensemble_soft_acc - single_acc),
            'confusion_matrix': ensemble_soft_cm.tolist()
        },
        'ensemble_hard': {
            'accuracy': float(ensemble_hard_acc),
            'f1_score': float(ensemble_hard_f1),
            'improvement': float(ensemble_hard_acc - single_acc),
            'confusion_matrix': ensemble_hard_cm.tolist()
        },
        'test_set_size': len(test_texts),
        'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_file = os.path.join(config['results_dir'], 'svm_evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 评估结果已保存到: {results_file}")
    
    # ==================== 总结 ====================
    print("\n" + "="*70)
    print("✅ SVM模型评估完成")
    print("="*70)
    
    print(f"\n📊 最终结果:")
    print(f"  单模型测试准确率: {single_acc*100:.2f}%")
    print(f"  集成模型（软投票）: {ensemble_soft_acc*100:.2f}% (+{(ensemble_soft_acc - single_acc)*100:.2f}%)")
    print(f"  集成模型（硬投票）: {ensemble_hard_acc*100:.2f}% (+{(ensemble_hard_acc - single_acc)*100:.2f}%)")
    
    print(f"\n💾 生成的文件:")
    print(f"  1. svm_single_cm.png - 单模型混淆矩阵")
    print(f"  2. svm_ensemble_soft_cm.png - 集成模型（软投票）混淆矩阵")
    print(f"  3. svm_ensemble_hard_cm.png - 集成模型（硬投票）混淆矩阵")
    print(f"  4. svm_performance_comparison.png - 性能对比图")
    print(f"  5. svm_evaluation_results.json - 详细评估结果")


if __name__ == "__main__":
    main()