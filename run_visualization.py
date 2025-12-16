
"""
机器学习模型数据可视化脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_visualizations():
    """创建可视化图表"""
    output_dir = Path('comprehensive_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # 颜色配置
    colors = {
        'BERT': '#FF6B6B',
        'LSTM': '#4ECDC4', 
        'Naive Bayes': '#45B7D1',
        'SVM': '#96CEB4'
    }
    
    print("开始生成可视化图表...")
    
    # 1. 加载数据并创建模型性能对比图
    models_data = []
    
    # BERT数据
    try:
        bert_ensemble = pd.read_csv('models/bert_model/models/bert_ensemble_performance.csv')
        models_data.append({
            'model': 'BERT',
            'accuracy': bert_ensemble['ensemble_soft_accuracy'].iloc[0],
            'f1': bert_ensemble['ensemble_soft_f1'].iloc[0],
            'color': colors['BERT']
        })
        print("✓ BERT数据加载完成")
    except Exception as e:
        print(f"Warning: BERT数据加载失败: {e}")
    
    # 朴素贝叶斯数据
    try:
        with open('nb_evaluation_results/nb_evaluation_results.json', 'r') as f:
            nb_data = json.load(f)
        models_data.append({
            'model': 'Naive Bayes',
            'accuracy': nb_data['metrics']['ensemble']['accuracy'],
            'f1': nb_data['metrics']['ensemble']['f1'],
            'color': colors['Naive Bayes']
        })
        print("✓ 朴素贝叶斯数据加载完成")
    except Exception as e:
        print(f"Warning: 朴素贝叶斯数据加载失败: {e}")
    
    # SVM数据
    try:
        with open('evaluation_results/svm_evaluation_results.json', 'r') as f:
            svm_data = json.load(f)
        models_data.append({
            'model': 'SVM',
            'accuracy': svm_data['ensemble_soft']['accuracy'],
            'f1': svm_data['ensemble_soft']['f1_score'],
            'color': colors['SVM']
        })
        print("✓ SVM数据加载完成")
    except Exception as e:
        print(f"Warning: SVM数据加载失败: {e}")
    
    # LSTM数据
    try:
        lstm_log = pd.read_csv('models/lstm_model/lstm_training_log_fold_0.csv')
        best_acc = lstm_log['val_accuracy'].max()
        best_f1 = lstm_log['val_f1'].max()
        models_data.append({
            'model': 'LSTM',
            'accuracy': best_acc,
            'f1': best_f1,
            'color': colors['LSTM']
        })
        print("✓ LSTM数据加载完成")
    except Exception as e:
        print(f"Warning: LSTM数据加载失败: {e}")
    
    if not models_data:
        print("错误: 没有成功加载任何模型数据")
        return
    
    # 创建性能对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = [data['model'] for data in models_data]
    accuracies = [data['accuracy'] for data in models_data]
    f1_scores = [data['f1'] for data in models_data]
    colors_list = [data['color'] for data in models_data]
    
    # 准确率对比
    bars1 = ax1.bar(models, accuracies, color=colors_list, alpha=0.8, edgecolor='black')
    ax1.set_title('模型准确率对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('准确率')
    ax1.set_ylim(0.8, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # F1分数对比
    bars2 = ax2.bar(models, f1_scores, color=colors_list, alpha=0.8, edgecolor='black')
    ax2.set_title('模型F1分数对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1分数')
    ax2.set_ylim(0.8, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{f1:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 模型性能对比图已保存")
    
    # 2. 创建训练曲线对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('模型训练过程对比', fontsize=16, fontweight='bold')
    
    # BERT训练曲线
    try:
        bert_history = pd.read_csv('models/bert_model/bert_training_history.csv')
        epochs = bert_history['epoch']
        
        axes[0,0].plot(epochs, bert_history['train_loss'], 
                      color=colors['BERT'], label='训练损失', linewidth=2)
        axes[0,0].plot(epochs, bert_history['val_loss'], 
                      color=colors['BERT'], label='验证损失', linestyle='--', linewidth=2)
        axes[0,0].set_title('BERT训练曲线')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
    except Exception as e:
        print(f"Warning: 无法绘制BERT训练曲线: {e}")
    
    # LSTM训练曲线
    try:
        lstm_log = pd.read_csv('models/lstm_model/lstm_training_log_fold_0.csv')
        epochs = lstm_log['epoch']
        
        axes[0,1].plot(epochs, lstm_log['train_loss'], 
                      color=colors['LSTM'], label='训练损失', linewidth=2)
        axes[0,1].plot(epochs, lstm_log['val_loss'], 
                      color=colors['LSTM'], label='验证损失', linestyle='--', linewidth=2)
        axes[0,1].set_title('LSTM训练曲线')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    except Exception as e:
        print(f"Warning: 无法绘制LSTM训练曲线: {e}")
    
    # 准确率对比
    try:
        bert_history = pd.read_csv('models/bert_model/bert_training_history.csv')
        epochs = bert_history['epoch']
        axes[1,0].plot(epochs, bert_history['train_acc'], 
                      color=colors['BERT'], label='BERT训练准确率', linewidth=2)
        axes[1,0].plot(epochs, bert_history['val_acc'], 
                      color=colors['BERT'], label='BERT验证准确率', linestyle='--', linewidth=2)
        
        lstm_log = pd.read_csv('models/lstm_model/lstm_training_log_fold_0.csv')
        epochs_lstm = lstm_log['epoch']
        axes[1,0].plot(epochs_lstm, lstm_log['train_accuracy'], 
                      color=colors['LSTM'], label='LSTM训练准确率', linewidth=2)
        axes[1,0].plot(epochs_lstm, lstm_log['val_accuracy'], 
                      color=colors['LSTM'], label='LSTM验证准确率', linestyle='--', linewidth=2)
        
        axes[1,0].set_title('准确率对比')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    except Exception as e:
        print(f"Warning: 无法绘制准确率对比图: {e}")
    
    # F1分数对比
    try:
        bert_history = pd.read_csv('models/bert_model/bert_training_history.csv')
        epochs = bert_history['epoch']
        axes[1,1].plot(epochs, bert_history['train_f1'], 
                      color=colors['BERT'], label='BERT训练F1', linewidth=2)
        axes[1,1].plot(epochs, bert_history['val_f1'], 
                      color=colors['BERT'], label='BERT验证F1', linestyle='--', linewidth=2)
        
        lstm_log = pd.read_csv('models/lstm_model/lstm_training_log_fold_0.csv')
        epochs_lstm = lstm_log['epoch']
        axes[1,1].plot(epochs_lstm, lstm_log['train_f1'], 
                      color=colors['LSTM'], label='LSTM训练F1', linewidth=2)
        axes[1,1].plot(epochs_lstm, lstm_log['val_f1'], 
                      color=colors['LSTM'], label='LSTM验证F1', linestyle='--', linewidth=2)
        
        axes[1,1].set_title('F1分数对比')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('F1 Score')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    except Exception as e:
        print(f"Warning: 无法绘制F1分数对比图: {e}")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 训练曲线对比图已保存")
    
    # 3. 创建集成方法对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('集成方法效果对比', fontsize=16, fontweight='bold')
    
    # BERT集成对比
    try:
        bert_data = pd.read_csv('models/bert_model/models/bert_ensemble_performance.csv')
        methods = ['单个模型', '软投票', '硬投票']
        accuracies = [
            bert_data['single_model_accuracy'].iloc[0],
            bert_data['ensemble_soft_accuracy'].iloc[0],
            bert_data['ensemble_hard_accuracy'].iloc[0]
        ]
        
        axes[0,0].bar(methods, accuracies, 
                     color=[colors['BERT'], '#FECA57', '#FECA57'],
                     alpha=0.8, edgecolor='black')
        axes[0,0].set_title('BERT集成方法对比')
        axes[0,0].set_ylabel('准确率')
        axes[0,0].set_ylim(0.85, 0.95)
        for i, v in enumerate(accuracies):
            axes[0,0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    except Exception as e:
        print(f"Warning: 无法绘制BERT集成对比图: {e}")
    
    # 朴素贝叶斯集成对比
    try:
        nb_data = json.load(open('nb_evaluation_results/nb_evaluation_results.json'))
        methods = ['单个模型', '集成模型']
        accuracies = [
            nb_data['metrics']['single']['accuracy'],
            nb_data['metrics']['ensemble']['accuracy']
        ]
        
        axes[0,1].bar(methods, accuracies, 
                     color=[colors['Naive Bayes'], '#FECA57'],
                     alpha=0.8, edgecolor='black')
        axes[0,1].set_title('朴素贝叶斯集成对比')
        axes[0,1].set_ylabel('准确率')
        axes[0,1].set_ylim(0.8, 0.9)
        for i, v in enumerate(accuracies):
            axes[0,1].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    except Exception as e:
        print(f"Warning: 无法绘制朴素贝叶斯集成对比图: {e}")
    
    # SVM集成对比
    try:
        svm_data = json.load(open('evaluation_results/svm_evaluation_results.json'))
        methods = ['单个模型', '软投票', '硬投票']
        accuracies = [
            svm_data['single_model']['accuracy'],
            svm_data['ensemble_soft']['accuracy'],
            svm_data['ensemble_hard']['accuracy']
        ]
        
        axes[1,0].bar(methods, accuracies, 
                     color=[colors['SVM'], '#FECA57', '#FECA57'],
                     alpha=0.8, edgecolor='black')
        axes[1,0].set_title('SVM集成方法对比')
        axes[1,0].set_ylabel('准确率')
        axes[1,0].set_ylim(0.8, 0.9)
        for i, v in enumerate(accuracies):
            axes[1,0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    except Exception as e:
        print(f"Warning: 无法绘制SVM集成对比图: {e}")
    
    # 集成改进效果总结
    improvements = []
    model_names = []
    
    try:
        bert_data = pd.read_csv('models/bert_model/models/bert_ensemble_performance.csv')
        improvements.append(bert_data['soft_improvement'].iloc[0])
        model_names.append('BERT')
    except:
        pass
    
    try:
        nb_data = json.load(open('nb_evaluation_results/nb_evaluation_results.json'))
        improvements.append(nb_data['performance_improvement']['accuracy_improvement'])
        model_names.append('Naive Bayes')
    except:
        pass
    
    try:
        svm_data = json.load(open('evaluation_results/svm_evaluation_results.json'))
        improvements.append(svm_data['ensemble_soft']['improvement'])
        model_names.append('SVM')
    except:
        pass
    
    if improvements:
        colors_imp = [colors[name] for name in model_names]
        bars = axes[1,1].bar(model_names, improvements, color=colors_imp, alpha=0.8, edgecolor='black')
        axes[1,1].set_title('集成方法改进效果')
        axes[1,1].set_ylabel('准确率提升')
        axes[1,1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(improvements):
            axes[1,1].text(i, v + 0.0001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 集成方法对比图已保存")
    
    print(f"\n所有可视化图表已保存到: {output_dir}")
    print("生成的文件:")
    for file in output_dir.glob('*.png'):
        print(f"  - {file.name}")

if __name__ == "__main__":
    create_visualizations()
