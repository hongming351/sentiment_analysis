
"""
结果可视化模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

class ResultsVisualizer:
    """结果可视化类"""
    def __init__(self, results):
        """
        初始化可视化器
        
        Args:
            results: 模型结果字典
        """
        self.results = results
        self.figures_dir = Path("results/performance_plots")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_comparison(self):
        """绘制模型对比图"""
        if not self.results:
            print("没有结果数据可供可视化")
            return
        
        # 提取数据
        models = list(self.results.keys())
        metrics = ['accuracy', 'f1_score', 'recall', 'precision']
        metric_names = ['准确率', 'F1分数', '召回率', '精确率']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('模型性能对比', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            # 提取该指标的数据
            values = []
            for model in models:
                if metric in self.results[model]:
                    values.append(self.results[model][metric])
                else:
                    values.append(0)
            
            # 绘制柱状图
            bars = ax.bar(models, values, color=colors)
            ax.set_title(f'{metric_name}对比')
            ax.set_ylabel(metric_name)
            ax.set_ylim([0, 1])
            
            # 在柱子上添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.figures_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存: {save_path}")
        plt.show()
        
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.results:
            print("没有结果数据可供可视化")
            return
        
        # 找到有训练历史的模型
        models_with_history = []
        for model_name, result in self.results.items():
            if 'history' in result and result['history']:
                models_with_history.append(model_name)
        
        if not models_with_history:
            print("没有找到训练历史数据")
            return
        
        # 创建子图
        n_models = len(models_with_history)
        fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('训练过程曲线', fontsize=16, fontweight='bold')
        
        for idx, model_name in enumerate(models_with_history):
            history = self.results[model_name]['history']
            
            # 损失曲线
            ax_loss = axes[idx, 0] if n_models > 1 else axes[0]
            ax_loss.plot(history['train_loss'], label='训练损失', marker='o')
            if 'val_loss' in history:
                ax_loss.plot(history['val_loss'], label='验证损失', marker='s')
            ax_loss.set_title(f'{model_name} - 损失曲线')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('损失')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            
            # 准确率曲线
            ax_acc = axes[idx, 1] if n_models > 1 else axes[1]
            ax_acc.plot(history['train_acc'], label='训练准确率', marker='o')
            if 'val_acc' in history:
                ax_acc.plot(history['val_acc'], label='验证准确率', marker='s')
            ax_acc.set_title(f'{model_name} - 准确率曲线')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('准确率')
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.figures_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线图已保存: {save_path}")
        plt.show()
        
    def plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        if not self.results:
            print("没有结果数据可供可视化")
            return
        
        from sklearn.metrics import confusion_matrix
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('混淆矩阵对比', fontsize=16, fontweight='bold')
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            if 'predictions' in result and 'labels' in result:
                cm = confusion_matrix(result['labels'], result['predictions'])
                
                ax = axes[idx]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(model_name)
                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.figures_dir / "confusion_matrices.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵图已保存: {save_path}")
        plt.show()
        
    def plot_inference_time_comparison(self):
        """绘制推理时间对比图"""
        if not self.results:
            print("没有结果数据可供可视化")
            return
        
        models = list(self.results.keys())
        inference_times = []
        
        for model in models:
            if 'inference_time' in self.results[model]:
                inference_times.append(self.results[model]['inference_time'])
            else:
                inference_times.append(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, inference_times, color=colors)
        
        ax.set_title('模型推理时间对比')
        ax.set_xlabel('模型')
        ax.set_ylabel('推理时间 (ms)')
        
        # 在柱子上添加数值标签
        for bar, time_val in zip(bars, inference_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{time_val:.2f}ms', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图像
        save_path = self.figures_dir / "inference_time_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"推理时间对比图已保存: {save_path}")
        plt.show()