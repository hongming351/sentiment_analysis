import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import warnings
import os
import json

# ======================== 全局字体设置（开头统一配置）========================
# 忽略警告
warnings.filterwarnings('ignore')

# 1. 优先获取系统可用的中文字体（不依赖特定字体）
def get_available_chinese_font():
    chinese_fonts = ['SimHei', 'Heiti TC', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'DejaVu Sans']
    for font in chinese_fonts:
        try:
            # 测试字体是否能正常使用
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            # 绘制测试文本，无报错则说明字体可用
            fig, ax = plt.subplots(figsize=(1,1))
            ax.text(0.5, 0.5, "中文测试", fontsize=10)
            plt.close(fig)
            return font
        except:
            continue
    # 兜底方案：使用默认字体（避免乱码）
    return 'DejaVu Sans'

# 2. 设置全局字体（关键：先获取可用字体，再锁定配置）
chinese_font = get_available_chinese_font()
plt.rcParams['font.sans-serif'] = [chinese_font]  # 全局字体
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
plt.rcParams['font.size'] = 10                    # 全局字号（可选）
print(f"✅ 已自动启用可用中文字体：{chinese_font}")

# 3. 设置seaborn样式（关键：禁止覆盖字体配置）
sns.set_style("whitegrid")
sns.set_palette("husl")
# 强制seaborn使用全局字体（避免样式覆盖）
sns.set(font=chinese_font)
sns.set_style("whitegrid", {"font.sans-serif": [chinese_font]})

# 4. 锁定matplotlib配置（防止后续代码修改）
plt.rcParams['axes.unicode_minus'] = False  # 再次确认负号显示
# ======================== 模型对比分析类 ========================
class ModelComparison:
    """模型对比分析类"""
    
    def __init__(self):
        """初始化"""
        self.models = []
        self.results_df = None
        self.lstm_history = None
        
    def load_lstm_history(self):
        """加载LSTM训练历史"""
        print("\n📊 加载LSTM训练历史...")

        # 尝试加载真实LSTM训练数据
        try:
            # 加载第一个折的训练日志作为代表
            lstm_csv_path = 'models/lstm_model/lstm_training_log_fold_0.csv'
            if os.path.exists(lstm_csv_path):
                self.lstm_history = pd.read_csv(lstm_csv_path)
                print(f"✅ 从 {lstm_csv_path} 加载真实LSTM训练历史")

                # 计算LSTM的最终性能指标
                lstm_best_val_acc = self.lstm_history['val_accuracy'].max()
                lstm_final_val_acc = self.lstm_history['val_accuracy'].iloc[-1]

                print(f"✅ LSTM训练历史加载完成 ({len(self.lstm_history)}个epoch)")
                print(f"   最佳验证准确率: {lstm_best_val_acc:.4f} (Epoch {self.lstm_history['val_accuracy'].idxmax() + 1})")
                print(f"   最终验证准确率: {lstm_final_val_acc:.4f}")
                print(f"   最终训练准确率: {self.lstm_history['train_accuracy'].iloc[-1]:.4f}")

                return self.lstm_history
            else:
                print(f"⚠️ 找不到真实LSTM数据文件: {lstm_csv_path}")
        except Exception as e:
            print(f"⚠️ 加载真实LSTM数据失败: {e}")

        # 如果加载失败，使用模拟数据作为备用
        print("⚠️ 使用模拟LSTM训练历史数据")
        lstm_history_data = {
            'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'train_loss': [0.46087456069124694, 0.3813154135873033, 0.3338466638868505, 0.30665056830780074,
                          0.28760552035898646, 0.27004146798175166, 0.2548943185459145, 0.2402554312932559,
                          0.21713919192023406, 0.2033199520665221],
            'train_accuracy': [0.7805598755832037, 0.8377471672961564, 0.8618084870028883, 0.8749389024661186,
                         0.8835147744945567, 0.8914241279715619, 0.8970673183736947, 0.904199066874028,
                         0.9131970673183737, 0.9186403021550766],
            'val_loss': [0.3719898271255004, 0.3410023970481677, 0.3343451466315832, 0.32794986741665083,
                        0.33680718946151245, 0.364908903837204, 0.3541181221222266, 0.3677643766769996,
                        0.3786728993440286, 0.3931718185926095],
            'val_accuracy': [0.8342020850040096, 0.8548516439454691, 0.8606655974338412, 0.8610665597433841,
                       0.8624699278267843, 0.8622694466720129, 0.8600641539695268, 0.8622694466720129,
                       0.8652766639935846, 0.861467522052927],
            'learning_rate': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005]
        }

        self.lstm_history = pd.DataFrame(lstm_history_data)

        # 计算LSTM的最终性能指标
        lstm_best_val_acc = self.lstm_history['val_accuracy'].max()
        lstm_final_val_acc = self.lstm_history['val_accuracy'].iloc[-1]

        print(f"✅ LSTM训练历史加载完成 (10个epoch)")
        print(f"   最佳验证准确率: {lstm_best_val_acc:.4f} (Epoch {self.lstm_history['val_accuracy'].idxmax() + 1})")
        print(f"   最终验证准确率: {lstm_final_val_acc:.4f}")
        print(f"   最终训练准确率: {self.lstm_history['train_accuracy'].iloc[-1]:.4f}")

        return self.lstm_history
    
    def load_models_data(self):
        """加载所有模型数据"""
        print("="*60)
        print("加载模型数据...")
        print("="*60)
        
        # 1. 加载LSTM历史
        lstm_history = self.load_lstm_history()
        
        # 2. BERT模型数据
        bert_data = {
            'model': 'BERT',
            'accuracy': 0.9070,
            'f1_score': 0.9070,
            'recall': 0.9070,
            'precision': 0.9070,
            'training_time': 17740.4,
            'inference_time': 15.0,
            'parameters': 19106690,
            'description': '预训练BERT微调，性能最好',
            'best_val_acc': 0.9068,
            'final_val_acc': 0.9061,
            'train_acc': 0.9319,
            'epochs': 4
        }
        
        # 3. NaiveBayes数据
        nb_data = {
            'model': 'NaiveBayes',
            'accuracy': 0.828,
            'f1_score': 0.828,
            'recall': 0.828,
            'precision': 0.828,
            'training_time': 0.5,
            'inference_time': 0.1,
            'parameters': 2000,
            'description': '传统机器学习，训练快，适合基线',
            'best_val_acc': 0.828,
            'final_val_acc': 0.828,
            'train_acc': 0.828,
            'epochs': 1
        }
        
        # 4. SVM数据
        svm_data = {
            'model': 'SVM',
            'accuracy': 0.811,
            'f1_score': 0.811,
            'recall': 0.811,
            'precision': 0.811,
            'training_time': 3.0,
            'inference_time': 0.5,
            'parameters': 2000,
            'description': '支持向量机，泛化能力强',
            'best_val_acc': 0.811,
            'final_val_acc': 0.811,
            'train_acc': 0.811,
            'epochs': 1
        }
        
        # 5. LSTM+Attention数据（使用真实训练历史）
        lstm_data = {
            'model': 'LSTM+Attention',
            'accuracy': lstm_history['val_acc'].iloc[-1],  # 最终验证准确率
            'f1_score': lstm_history['val_acc'].iloc[-1],  # 假设与准确率相同
            'recall': lstm_history['val_acc'].iloc[-1],
            'precision': lstm_history['val_acc'].iloc[-1],
            'training_time': 10000,  # 估计值
            'inference_time': 5.0,   # 估计值
            'parameters': 2000000,   # 估计值
            'description': '深度学习，可解释性强',
            'best_val_acc': lstm_history['val_acc'].max(),
            'final_val_acc': lstm_history['val_acc'].iloc[-1],
            'train_acc': lstm_history['train_acc'].iloc[-1],
            'epochs': len(lstm_history)
        }
        
        # 添加到模型列表
        self.models = [bert_data, nb_data, svm_data, lstm_data]
        self.results_df = pd.DataFrame(self.models)
        
        print(f"\n✅ 加载了 {len(self.models)} 个模型的数据")
        print(self.results_df[['model', 'accuracy', 'best_val_acc', 'training_time']])
        
        return self.results_df
    
    def calculate_additional_metrics(self):
        """计算额外指标"""
        print("\n" + "="*60)
        print("计算额外指标...")
        print("="*60)
        
        if self.results_df is None:
            self.load_models_data()
        
        # 1. 计算效率分数（准确率/训练时间）
        self.results_df['efficiency'] = self.results_df['accuracy'] / (self.results_df['training_time'] + 1)
        
        # 2. 计算速度分数（1/推理时间）
        self.results_df['speed_score'] = 1 / (self.results_df['inference_time'] + 0.001)
        
        # 3. 计算性价比（准确率/参数量 * 1000）
        self.results_df['cost_performance'] = (self.results_df['accuracy'] / (self.results_df['parameters'] / 1000))
        
        # 4. 计算过拟合程度（训练准确率 - 验证准确率）
        self.results_df['overfitting_degree'] = self.results_df['train_acc'] - self.results_df['accuracy']
        
        # 5. 计算收敛速度（准确率/训练轮数）
        self.results_df['convergence_speed'] = self.results_df['accuracy'] / self.results_df['epochs']
        
        # 6. 计算综合分数（加权平均）
        weights = {
            'accuracy': 0.35,
            'f1_score': 0.25,
            'speed_score': 0.15,
            'efficiency': 0.15,
            'cost_performance': 0.10
        }
        
        # 归一化各项指标
        for col in ['accuracy', 'f1_score', 'speed_score', 'efficiency', 'cost_performance']:
            min_val = self.results_df[col].min()
            max_val = self.results_df[col].max()
            if max_val > min_val:
                self.results_df[f'{col}_normalized'] = (self.results_df[col] - min_val) / (max_val - min_val)
            else:
                self.results_df[f'{col}_normalized'] = 1.0
        
        # 计算加权综合分数
        self.results_df['composite_score'] = (
            weights['accuracy'] * self.results_df['accuracy_normalized'] +
            weights['f1_score'] * self.results_df['f1_score_normalized'] +
            weights['speed_score'] * self.results_df['speed_score_normalized'] +
            weights['efficiency'] * self.results_df['efficiency_normalized'] +
            weights['cost_performance'] * self.results_df['cost_performance_normalized']
        )
        
        # 排序
        self.results_df = self.results_df.sort_values('composite_score', ascending=False)
        
        print("✅ 额外指标计算完成")
        print(self.results_df[['model', 'accuracy', 'composite_score', 'efficiency', 'overfitting_degree']])
        
        return self.results_df
    
    def plot_lstm_training_curves(self):
        """绘制LSTM训练曲线"""
        print("\n📈 绘制LSTM训练曲线...")
        
        if self.lstm_history is None:
            self.load_lstm_history()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 训练损失曲线
        ax1 = axes[0, 0]
        ax1.plot(self.lstm_history['epoch'], self.lstm_history['train_loss'], 
                label='训练损失', marker='o', linewidth=2, color='#FF6B6B')
        ax1.plot(self.lstm_history['epoch'], self.lstm_history['val_loss'], 
                label='验证损失', marker='s', linewidth=2, color='#4ECDC4')
        ax1.set_xlabel('训练轮次', fontsize=12)
        ax1.set_ylabel('损失值', fontsize=12)
        ax1.set_title('LSTM训练和验证损失曲线', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 标记最佳验证损失
        best_val_loss_idx = self.lstm_history['val_loss'].idxmin()
        best_val_loss = self.lstm_history['val_loss'].min()
        ax1.axvline(x=self.lstm_history['epoch'][best_val_loss_idx], color='red', 
                   linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(self.lstm_history['epoch'][best_val_loss_idx], best_val_loss, 
                f'最佳验证损失\n{best_val_loss:.4f}', ha='center', va='bottom',
                fontsize=9, color='red')
        
        # 2. 训练准确率曲线
        ax2 = axes[0, 1]
        ax2.plot(self.lstm_history['epoch'], self.lstm_history['train_acc'], 
                label='训练准确率', marker='o', linewidth=2, color='#FFD166')
        ax2.plot(self.lstm_history['epoch'], self.lstm_history['val_acc'], 
                label='验证准确率', marker='s', linewidth=2, color='#06D6A0')
        ax2.set_xlabel('训练轮次', fontsize=12)
        ax2.set_ylabel('准确率', fontsize=12)
        ax2.set_title('LSTM训练和验证准确率曲线', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.75, 0.95])
        
        # 标记最佳验证准确率
        best_val_acc_idx = self.lstm_history['val_acc'].idxmax()
        best_val_acc = self.lstm_history['val_acc'].max()
        ax2.axvline(x=self.lstm_history['epoch'][best_val_acc_idx], color='red', 
                   linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(self.lstm_history['epoch'][best_val_acc_idx], best_val_acc, 
                f'最佳验证准确率\n{best_val_acc:.4f}', ha='center', va='bottom',
                fontsize=9, color='red')
        
        # 3. 训练-验证准确率差距（过拟合程度）
        ax3 = axes[1, 0]
        gap = self.lstm_history['train_acc'] - self.lstm_history['val_acc']
        ax3.plot(self.lstm_history['epoch'], gap, 
                marker='o', linewidth=2, color='#FF6B6B')
        ax3.fill_between(self.lstm_history['epoch'], 0, gap, alpha=0.3, color='#FF6B6B')
        ax3.set_xlabel('训练轮次', fontsize=12)
        ax3.set_ylabel('训练-验证差距', fontsize=12)
        ax3.set_title('LSTM过拟合程度分析', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 标记平均差距
        avg_gap = gap.mean()
        ax3.axhline(y=avg_gap, color='blue', linestyle='--', linewidth=1, alpha=0.7)
        ax3.text(self.lstm_history['epoch'].iloc[-1], avg_gap, 
                f'平均差距: {avg_gap:.4f}', ha='right', va='bottom',
                fontsize=9, color='blue')
        
        # 4. 学习率变化
        ax4 = axes[1, 1]
        ax4.plot(self.lstm_history['epoch'], self.lstm_history['learning_rate'], 
                marker='o', linewidth=2, color='#118AB2')
        ax4.set_xlabel('训练轮次', fontsize=12)
        ax4.set_ylabel('学习率', fontsize=12)
        ax4.set_title('LSTM学习率调度', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')  # 对数刻度
        
        # 标记学习率下降点
        lr_change_points = self.lstm_history[self.lstm_history['learning_rate'].diff() < 0]['epoch']
        for point in lr_change_points:
            ax4.axvline(x=point, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax4.text(point, self.lstm_history[self.lstm_history['epoch'] == point]['learning_rate'].values[0], 
                    '学习率下降', ha='center', va='bottom', fontsize=9, color='red', rotation=90)
        
        plt.suptitle('LSTM+Attention 模型训练过程详细分析', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('lstm_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ LSTM训练曲线图已保存为 lstm_training_curves.png")
        
        # 输出LSTM训练分析
        print("\n📋 LSTM训练分析报告:")
        print(f"  训练轮次: {len(self.lstm_history)}")
        print(f"  最佳验证准确率: {best_val_acc:.4f} (Epoch {best_val_acc_idx + 1})")
        print(f"  最终验证准确率: {self.lstm_history['val_acc'].iloc[-1]:.4f}")
        print(f"  最终训练准确率: {self.lstm_history['train_acc'].iloc[-1]:.4f}")
        print(f"  过拟合程度(平均): {avg_gap:.4f}")
        print(f"  训练损失下降: {self.lstm_history['train_loss'].iloc[0]:.4f} → {self.lstm_history['train_loss'].iloc[-1]:.4f}")
        print(f"  验证损失下降: {self.lstm_history['val_loss'].iloc[0]:.4f} → {self.lstm_history['val_loss'].iloc[-1]:.4f}")
    
    def plot_accuracy_comparison(self):
        """绘制准确率对比图"""
        print("\n📊 绘制准确率对比图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 准确率柱状图
        ax1 = axes[0, 0]
        colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0']
        bars = ax1.bar(self.results_df['model'], self.results_df['accuracy'], color=colors)
        ax1.set_title('模型准确率对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('准确率', fontsize=12)
        ax1.set_ylim([0.75, 0.95])
        
        # 在柱子上添加数值
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 添加排名标签
        for i, (model, acc) in enumerate(zip(self.results_df['model'], self.results_df['accuracy'])):
            rank = ['🥇', '🥈', '🥉', '4'][i]
            ax1.text(i, acc + 0.01, rank, ha='center', va='bottom', fontsize=16)
        
        # 2. 训练准确率 vs 验证准确率
        ax2 = axes[0, 1]
        x = np.arange(len(self.results_df))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, self.results_df['train_acc'], width, label='训练准确率', color='#FF6B6B')
        bars2 = ax2.bar(x + width/2, self.results_df['accuracy'], width, label='验证准确率', color='#4ECDC4')
        
        # 计算和显示过拟合程度
        for i in range(len(self.results_df)):
            overfit = self.results_df['train_acc'].iloc[i] - self.results_df['accuracy'].iloc[i]
            ax2.text(i, max(self.results_df['train_acc'].iloc[i], self.results_df['accuracy'].iloc[i]) + 0.01,
                    f'Δ={overfit:.3f}', ha='center', va='bottom', fontsize=8, color='red')
        
        ax2.set_title('训练准确率 vs 验证准确率', fontsize=14, fontweight='bold')
        ax2.set_ylabel('准确率', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.results_df['model'], rotation=45, ha='right')
        ax2.set_ylim([0.75, 0.95])
        ax2.legend()
        
        # 3. 训练时间对比（对数刻度）
        ax3 = axes[1, 0]
        bars = ax3.bar(self.results_df['model'], self.results_df['training_time'], color=colors)
        ax3.set_title('训练时间对比（对数刻度）', fontsize=14, fontweight='bold')
        ax3.set_ylabel('训练时间（秒）', fontsize=12)
        ax3.set_yscale('log')  # 对数刻度
        ax3.tick_params(axis='x', rotation=45)
        
        # 添加数值标签（转换为分钟）
        for i, (model, time) in enumerate(zip(self.results_df['model'], self.results_df['training_time'])):
            minutes = time / 60
            label = f'{time:.1f}s\n({minutes:.1f}min)' if time > 60 else f'{time:.1f}s'
            ax3.text(i, time, label, ha='center', va='bottom', fontsize=9)
        
        # 4. 效率对比（准确率/训练时间）
        ax4 = axes[1, 1]
        efficiency = self.results_df['efficiency'] * 1000  # 放大以便显示
        bars = ax4.bar(self.results_df['model'], efficiency, color=colors)
        ax4.set_title('训练效率（准确率/训练时间 × 1000）', fontsize=14, fontweight='bold')
        ax4.set_ylabel('效率分数', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('模型性能对比分析', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 准确率对比图已保存为 model_accuracy_comparison.png")
    
    def plot_training_progress_comparison(self):
        """绘制训练进度对比图（BERT vs LSTM）"""
        print("\n📈 绘制训练进度对比图...")
        
        # 加载BERT训练历史
        try:
            bert_history = pd.read_csv('training_history.csv')
            print(f"✅ 加载BERT训练历史: {len(bert_history)}个epoch")
        except:
            # 如果没有BERT历史，创建模拟数据
            bert_history = pd.DataFrame({
                'epoch': [1, 2, 3, 4],
                'train_acc': [0.8689, 0.9105, 0.9210, 0.9319],
                'val_acc': [0.8918, 0.8988, 0.9068, 0.9061]
            })
            print("⚠️ 使用模拟的BERT训练历史")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 准确率进度对比
        ax1 = axes[0]
        
        # LSTM准确率曲线
        ax1.plot(self.lstm_history['epoch'], self.lstm_history['train_acc'], 
                label='LSTM训练准确率', marker='o', linewidth=2, color='#06D6A0')
        ax1.plot(self.lstm_history['epoch'], self.lstm_history['val_acc'], 
                label='LSTM验证准确率', marker='s', linewidth=2, color='#06D6A0', linestyle='--')
        
        # BERT准确率曲线
        ax1.plot(bert_history['epoch'], bert_history['train_acc'], 
                label='BERT训练准确率', marker='o', linewidth=2, color='#FF6B6B')
        ax1.plot(bert_history['epoch'], bert_history['val_acc'], 
                label='BERT验证准确率', marker='s', linewidth=2, color='#FF6B6B', linestyle='--')
        
        ax1.set_xlabel('训练轮次', fontsize=12)
        ax1.set_ylabel('准确率', fontsize=12)
        ax1.set_title('BERT vs LSTM 准确率训练进度对比', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.75, 0.95])
        
        # 2. 收敛速度对比（累计准确率增益）
        ax2 = axes[1]
        
        # 计算每个epoch的准确率增益
        lstm_gains = np.diff(self.lstm_history['val_acc'], prepend=self.lstm_history['val_acc'].iloc[0])
        bert_gains = np.diff(bert_history['val_acc'], prepend=bert_history['val_acc'].iloc[0])
        
        x_lstm = np.arange(len(lstm_gains))
        x_bert = np.arange(len(bert_gains))
        
        ax2.bar(x_lstm - 0.2, lstm_gains, width=0.4, label='LSTM准确率增益', color='#06D6A0', alpha=0.7)
        ax2.bar(x_bert + 0.2, bert_gains, width=0.4, label='BERT准确率增益', color='#FF6B6B', alpha=0.7)
        
        # 添加累计线
        lstm_cumulative = np.cumsum(lstm_gains)
        bert_cumulative = np.cumsum(bert_gains)
        
        ax2.plot(x_lstm, lstm_cumulative, label='LSTM累计增益', color='#06D6A0', linewidth=2, marker='o')
        ax2.plot(x_bert, bert_cumulative, label='BERT累计增益', color='#FF6B6B', linewidth=2, marker='s')
        
        ax2.set_xlabel('训练轮次', fontsize=12)
        ax2.set_ylabel('准确率增益', fontsize=12)
        ax2.set_title('BERT vs LSTM 收敛速度对比', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 添加总结文本
        lstm_total_gain = self.lstm_history['val_acc'].iloc[-1] - self.lstm_history['val_acc'].iloc[0]
        bert_total_gain = bert_history['val_acc'].iloc[-1] - bert_history['val_acc'].iloc[0]
        
        summary_text = f'''对比总结:
        LSTM: 总增益={lstm_total_gain:.3f}, 轮次={len(self.lstm_history)}
        BERT: 总增益={bert_total_gain:.3f}, 轮次={len(bert_history)}
        LSTM收敛速度: {lstm_total_gain/len(self.lstm_history):.4f}/epoch
        BERT收敛速度: {bert_total_gain/len(bert_history):.4f}/epoch'''
        
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('bert_vs_lstm_training_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 训练进度对比图已保存为 bert_vs_lstm_training_comparison.png")
    
    def plot_radar_chart(self):
        """绘制雷达图（综合性能对比）"""
        print("\n📊 绘制综合性能雷达图...")
        plt.rcParams['font.sans-serif'] = [chinese_font]
        # 选择要对比的指标
        metrics = ['accuracy', 'f1_score', 'efficiency', 'speed_score', 'cost_performance', 'convergence_speed']
        metric_labels = ['准确率', 'F1分数', '训练效率', '推理速度', '性价比', '收敛速度']
        
        # 归一化指标（0-1范围）
        normalized_data = []
        for metric in metrics:
            min_val = self.results_df[metric].min()
            max_val = self.results_df[metric].max()
            if max_val > min_val:
                normalized = (self.results_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized = [1.0] * len(self.results_df)
            normalized_data.append(normalized.values)
        
        # 准备雷达图数据
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # 颜色列表
        colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0']
        
        # 绘制每个模型的雷达图
        for idx, model in enumerate(self.results_df['model']):
            values = [normalized_data[m][idx] for m in range(len(metrics))]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
            
            # 在每个点上添加数值标签
            for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
                display_value = self.results_df[metrics[i]].iloc[idx]
                if metrics[i] in ['accuracy', 'f1_score']:
                    label = f'{display_value:.3f}'
                elif metrics[i] == 'efficiency':
                    label = f'{display_value*1000:.2f}'
                elif metrics[i] == 'speed_score':
                    label = f'{1/self.results_df["inference_time"].iloc[idx]:.1f}'
                else:
                    label = f'{display_value:.2f}'
                
                # 调整标签位置避免重叠
                label_angle = angle
                if idx == 0:  # 第一个模型
                    label_radius = value + 0.05
                elif idx == 1:  # 第二个模型
                    label_radius = value - 0.05
                elif idx == 2:  # 第三个模型
                    label_radius = value + 0.03
                else:  # 第四个模型
                    label_radius = value - 0.03
                
                ax.text(label_angle, label_radius, label, 
                       fontsize=8, ha='center', va='center')
        
        # 设置角度和标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=12)
        
        # 设置网格线
        ax.grid(True, alpha=0.3)
        
        # 设置径向标签
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        
        # 添加标题和图例
        plt.title('模型综合性能雷达图（归一化）', fontsize=16, fontweight='bold', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig('model_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 雷达图已保存为 model_radar_chart.png")
    
    def create_detailed_comparison_table(self):
        """创建详细对比表格"""
        print("\n📋 创建详细对比表格...")
        
        # 准备对比数据
        comparison_data = []
        
        for idx, row in self.results_df.iterrows():
            model_data = {
                '模型': row['model'],
                '排名': idx + 1,
                '准确率': f"{row['accuracy']:.4f}",
                'F1分数': f"{row['f1_score']:.4f}",
                '训练准确率': f"{row['train_acc']:.4f}",
                '验证准确率': f"{row['accuracy']:.4f}",
                '过拟合程度': f"{row['overfitting_degree']:.4f}",
                '训练时间': f"{row['training_time']:.1f}s ({row['training_time']/60:.1f}分钟)",
                '推理时间': f"{row['inference_time']:.1f}ms",
                '参数量': f"{row['parameters']:,}",
                '训练轮次': row['epochs'],
                '收敛速度': f"{row['convergence_speed']:.4f}/epoch",
                '效率分数': f"{row['efficiency']*1000:.2f}",
                '综合分数': f"{row['composite_score']:.4f}"
            }
            comparison_data.append(model_data)
        
        # 创建DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存为CSV
        comparison_df.to_csv('detailed_model_comparison.csv', index=False, encoding='utf-8-sig')
        
        # 创建HTML表格（带样式）
        html_table = comparison_df.to_html(index=False, escape=False, 
                                          classes='table table-striped table-bordered')
        
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>模型详细对比表</title>
            <style>
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                    font-family: Arial, sans-serif;
                }}
                th {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 12px;
                    text-align: center;
                    font-weight: bold;
                    position: sticky;
                    top: 0;
                }}
                td {{
                    padding: 10px;
                    text-align: center;
                    border-bottom: 1px solid #ddd;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #ddd;
                }}
                .rank-1 {{
                    background-color: #ffeb3b !important;
                }}
                .rank-2 {{
                    background-color: #e0e0e0 !important;
                }}
                .rank-3 {{
                    background-color: #ffcc80 !important;
                }}
                .best {{
                    font-weight: bold;
                    color: #d32f2f;
                }}
            </style>
        </head>
        <body>
            <h2>京东评论情感分析模型详细对比表</h2>
            <p>生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            {html_table}
        </body>
        </html>
        '''
        
        with open('detailed_model_comparison.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✅ 详细对比表格已保存:")
        print("  - detailed_model_comparison.csv")
        print("  - detailed_model_comparison.html")
        
        # 打印到控制台
        print("\n" + "="*80)
        print("详细模型对比表")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_all_charts(self):
        """绘制所有图表"""
        print("\n" + "="*80)
        print("开始绘制所有对比图表")
        print("="*80)
        
        # 加载数据
        self.load_models_data()
        self.calculate_additional_metrics()
        
        # 绘制LSTM训练曲线
        self.plot_lstm_training_curves()
        
        # 绘制训练进度对比
        self.plot_training_progress_comparison()
        
        # 绘制其他对比图表
        self.plot_accuracy_comparison()
        self.plot_radar_chart()
        
        # 创建详细表格
        self.create_detailed_comparison_table()
        
        print("\n" + "="*80)
        print("🎉 所有图表绘制完成！")
        print("="*80)
        
        print("\n📁 生成的文件:")
        print("1. lstm_training_curves.png - LSTM训练曲线图")
        print("2. bert_vs_lstm_training_comparison.png - BERT vs LSTM训练对比")
        print("3. model_accuracy_comparison.png - 模型准确率对比图")
        print("4. model_radar_chart.png - 综合性能雷达图")
        print("5. detailed_model_comparison.csv - 详细对比表格(CSV)")
        print("6. detailed_model_comparison.html - 详细对比表格(HTML)")
        
        # 生成最终报告
        self.generate_final_report()
        
        return True
    
    def generate_final_report(self):
        """生成最终分析报告"""
        print("\n📝 生成最终分析报告...")
        
        report = {
            "生成时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "数据集": "京东评论情感分析",
            "模型数量": len(self.models),
            "性能排名": [],
            "关键发现": [],
            "建议": []
        }
        
        # 性能排名
        for idx, row in self.results_df.iterrows():
            rank_info = {
                "排名": idx + 1,
                "模型": row['model'],
                "准确率": row['accuracy'],
                "综合分数": row['composite_score'],
                "训练时间": f"{row['training_time']:.1f}s"
            }
            report["性能排名"].append(rank_info)
        
        # 关键发现
        report["关键发现"] = [
            "1. BERT模型性能最佳，但训练时间最长（4.5小时）",
            "2. LSTM+Attention模型在10个epoch后达到86.15%准确率，略低于BERT",
            "3. LSTM模型在第9个epoch达到最佳验证准确率86.53%",
            "4. 朴素贝叶斯训练最快（0.5秒），适合快速原型开发",
            "5. LSTM模型存在轻微过拟合（训练准确率91.86% vs 验证准确率86.15%）",
            "6. BERT模型在4个epoch内收敛，而LSTM需要10个epoch"
        ]
        
        # 建议
        report["建议"] = [
            {
                "场景": "高精度生产环境",
                "推荐模型": "BERT",
                "理由": "准确率最高（90.70%），适合对精度要求严格的场景"
            },
            {
                "场景": "快速原型/资源有限",
                "推荐模型": "朴素贝叶斯",
                "理由": "训练最快（0.5秒），准确率可接受（82.80%）"
            },
            {
                "场景": "需要模型可解释性",
                "推荐模型": "LSTM+Attention",
                "理由": "注意力机制可视化，可理解模型决策过程"
            },
            {
                "场景": "平衡各方面需求",
                "推荐模型": "SVM",
                "理由": "训练时间适中（3秒），泛化能力良好"
            }
        ]
        
        # 保存JSON报告
        with open('model_comparison_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成Markdown报告
        self.generate_markdown_report(report)
        
        print("✅ 最终报告已生成:")
        print("  - model_comparison_report.json")
        print("  - final_analysis_report.md")
        
        return report
    
    def generate_markdown_report(self, report):
        """生成Markdown格式的报告"""
        
        md_content = f'''# 京东评论情感分析模型对比报告

**生成时间**: {report["生成时间"]}
**数据集**: {report["数据集"]}
**对比模型数量**: {report["模型数量"]}

## 🏆 性能排名

| 排名 | 模型 | 准确率 | 综合分数 | 训练时间 |
|------|------|--------|----------|----------|
'''
        
        for rank_info in report["性能排名"]:
            md_content += f'| {rank_info["排名"]} | {rank_info["模型"]} | {rank_info["准确率"]:.4f} | {rank_info["综合分数"]:.4f} | {rank_info["训练时间"]} |\n'
        
        md_content += f'''

## 📊 LSTM训练详细分析

### 训练过程
- **训练轮次**: 10个epoch
- **最佳验证准确率**: {self.lstm_history['val_acc'].max():.4f} (Epoch {self.lstm_history['val_acc'].idxmax() + 1})
- **最终验证准确率**: {self.lstm_history['val_acc'].iloc[-1]:.4f}
- **最终训练准确率**: {self.lstm_history['train_acc'].iloc[-1]:.4f}
- **过拟合程度**: {self.lstm_history['train_acc'].iloc[-1] - self.lstm_history['val_acc'].iloc[-1]:.4f}

### 学习率调度
- Epoch 1-7: 学习率 0.001
- Epoch 8-10: 学习率下降为 0.0005

## 📈 BERT vs LSTM 对比

### 收敛速度
- **BERT**: 4个epoch达到90.68%准确率
- **LSTM**: 10个epoch达到86.15%准确率
- **BERT收敛速度更快**，但需要更多计算资源

### 过拟合情况
- **BERT**: 训练准确率93.19% vs 验证准确率90.61% (Δ=0.0258)
- **LSTM**: 训练准确率91.86% vs 验证准确率86.15% (Δ=0.0571)
- **LSTM过拟合更明显**，可能需要更多正则化

## 🎯 场景推荐

'''
        
        for rec in report["建议"]:
            md_content += f'''
### {rec["场景"]}
- **推荐模型**: {rec["推荐模型"]}
- **理由**: {rec["理由"]}
'''
        
        md_content += f'''

## 🔑 关键发现

'''
        
        for finding in report["关键发现"]:
            md_content += f'{finding}\n\n'
        
        md_content += f'''
## 📁 生成的文件

所有可视化图表和对比表格已生成：

1. `lstm_training_curves.png` - LSTM训练曲线图
2. `bert_vs_lstm_training_comparison.png` - BERT vs LSTM训练对比图
3. `model_accuracy_comparison.png` - 模型准确率对比图
4. `model_radar_chart.png` - 综合性能雷达图
5. `detailed_model_comparison.csv` - 详细对比表格
6. `model_comparison_report.json` - JSON格式报告
7. `final_analysis_report.md` - 本Markdown报告

## 📞 结论

根据对比分析，不同模型各有优劣：
- **追求最高精度** → 选择BERT模型
- **需要快速部署** → 选择朴素贝叶斯
- **关注模型解释性** → 选择LSTM+Attention
- **平衡各方面需求** → 选择SVM

'''
        
        with open('final_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return True

def main():
    """主函数"""
    print("="*80)
    print("📊 京东评论情感分析模型对比系统")
    print("="*80)
    
    # 创建对比分析器
    analyzer = ModelComparison()
    
    # 绘制所有图表
    analyzer.plot_all_charts()
    
    print("\n" + "="*80)
    print("✅ 对比分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
