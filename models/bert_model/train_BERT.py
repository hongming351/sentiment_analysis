import pandas as pd
import numpy as np
import torch
import re
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import random
import time
import warnings
import os
import json
from tqdm import tqdm
import matplotlib
from collections import Counter

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STXihei']
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

def print_gpu_memory_usage(device=None):
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        cached = torch.cuda.memory_reserved(device) / 1024**2
        
        print(f"GPU内存使用: {allocated:.2f} MB (已分配) / {cached:.2f} MB (缓存)")
        return allocated, cached
    else:
        print("未检测到GPU")
        return 0, 0

def clear_gpu_cache():
    """清空GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清空GPU缓存")

# ==================== 1. 配置参数 ====================
class Config:
    # 数据路径
    data_dir = "data"
    
    # 模型参数
    model_path = "bert-base-chinese"  # 使用官方预训练模型
    max_length = 128
    num_labels = 2
    
    # 训练参数
    batch_size = 16
    num_epochs = 10  # 减少epoch数以加快训练
    learning_rate = 2e-5
    warmup_ratio = 0.1
    weight_decay = 0.01
    patience = 3  # 减少早停耐心值
    
    # 交叉验证
    n_folds = 5
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 随机种子
    seed = 42
    
    # 可视化目录
    viz_dir = 'bert_visualizations'
    
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

config = Config()
config.set_seed()

# 创建可视化目录
os.makedirs(config.viz_dir, exist_ok=True)

# ==================== 2. 数据清洗和预处理 ====================
def clean_text(text):
    """清洗文本"""
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：,.!?;\'"、]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_data(df, text_col='sentence', label_col='label'):
    """清洗数据"""
    print(f"原始数据大小: {len(df)}")
    
    # 备份原始数据
    df_original = df.copy()
    
    # 处理NaN值
    df = df.dropna(subset=[text_col, label_col]).copy()
    
    # 清理文本
    df[text_col] = df[text_col].apply(clean_text)
    
    # 确保标签为整数
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    
    # 移除空文本
    df = df[df[text_col].str.len() > 0].copy()
    
    print(f"清洗后数据大小: {len(df)}")
    print(f"移除的行数: {len(df_original) - len(df)}")
    
    return df

# ==================== 3. 数据集类 ====================
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
        
        return item

# ==================== 4. 可视化函数 ====================
def plot_training_history(history, fold_idx, save_dir=config.viz_dir):
    """绘制单折训练历史"""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'BERT模型 - 第{fold_idx+1}折训练历史', fontsize=16, fontweight='bold')
    
    # 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1-score曲线
    axes[0, 2].plot(epochs, history['train_f1'], 'b-', label='训练F1', linewidth=2)
    axes[0, 2].plot(epochs, history['val_f1'], 'r-', label='验证F1', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1-score')
    axes[0, 2].set_title('F1-score曲线')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 精确率曲线
    axes[1, 0].plot(epochs, history['train_precision'], 'b-', label='训练精确率', linewidth=2)
    axes[1, 0].plot(epochs, history['val_precision'], 'r-', label='验证精确率', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('精确率')
    axes[1, 0].set_title('精确率曲线')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 召回率曲线
    axes[1, 1].plot(epochs, history['train_recall'], 'b-', label='训练召回率', linewidth=2)
    axes[1, 1].plot(epochs, history['val_recall'], 'r-', label='验证召回率', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('召回率')
    axes[1, 1].set_title('召回率曲线')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 学习率曲线
    if 'learning_rate' in history:
        axes[1, 2].plot(epochs, history['learning_rate'], 'g-', label='学习率', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('学习率')
        axes[1, 2].set_title('学习率变化')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'bert_training_history_fold_{fold_idx}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ BERT第{fold_idx+1}折训练历史图已保存")

def plot_confusion_matrix_comprehensive(y_true, y_pred, fold_idx=None, model_name=None, save_dir=config.viz_dir):
    """绘制完整的混淆矩阵"""
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    classes = ['负面', '正面']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 热力图混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_xlabel('预测标签')
    axes[0].set_ylabel('真实标签')
    title = '混淆矩阵'
    if fold_idx is not None:
        title += f' (第{fold_idx+1}折)'
    if model_name:
        title += f' ({model_name})'
    axes[0].set_title(title)
    
    # 2. 百分比混淆矩阵
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Oranges',
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_xlabel('预测标签')
    axes[1].set_ylabel('真实标签')
    axes[1].set_title('混淆矩阵（百分比）')
    
    plt.tight_layout()
    filename = 'bert_confusion_matrix'
    if fold_idx is not None:
        filename += f'_fold_{fold_idx}'
    if model_name:
        filename += f'_{model_name}'
    filename += '.png'
    
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ BERT混淆矩阵已保存: {filename}")

# ==================== 5. 训练和评估函数 ====================
def train_epoch_bert(model, dataloader, optimizer, scheduler, device, class_weights=None):
    """BERT训练一个epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='训练', leave=False)
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # 如果有类别权重，调整损失
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            logits = outputs.logits
            loss = loss_fct(logits.view(-1, config.num_labels), labels.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # 记录
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy()
        all_predictions.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # 计算epoch指标
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
    epoch_precision = precision_score(all_labels, all_predictions, average='macro')
    epoch_recall = recall_score(all_labels, all_predictions, average='macro')
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'f1': epoch_f1,
        'precision': epoch_precision,
        'recall': epoch_recall
    }

def evaluate_epoch_bert(model, dataloader, device):
    """BERT评估一个epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='评估', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_predictions.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算epoch指标
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
    epoch_precision = precision_score(all_labels, all_predictions, average='macro')
    epoch_recall = recall_score(all_labels, all_predictions, average='macro')
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'f1': epoch_f1,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }

# ==================== 6. 单折训练函数 ====================
def train_single_fold_bert(fold_idx, train_df, val_df, tokenizer, device, n_epochs=10):
    """训练单个折的BERT模型"""
    print(f"\n{'='*70}")
    print(f"📊 BERT模型 - 第 {fold_idx+1}/5 折训练")
    print(f"📱 使用设备: {device}")
    print(f"{'='*70}")
    
    # 设置GPU优化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU内存使用情况: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
    
    # 创建数据集
    train_dataset = BERTDataset(train_df['sentence'].values, train_df['label'].values, tokenizer, config.max_length)
    val_dataset = BERTDataset(val_df['sentence'].values, val_df['label'].values, tokenizer, config.max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 创建模型
    print("创建BERT模型中...")
    model = BertForSequenceClassification.from_pretrained(
        config.model_path,
        num_labels=config.num_labels
    )
    
    # 将模型移动到设备
    model.to(device)
    
    # 如果是多GPU训练
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)
    
    # 计算类别权重
    class_counts = Counter(train_df['label'].values)
    total_samples = len(train_df)
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts.values()]
    
    # 创建优化器和调度器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps
    )
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'train_precision': [],
        'train_recall': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'learning_rate': []
    }
    
    # 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # 训练
        train_metrics = train_epoch_bert(model, train_loader, optimizer, scheduler, device, class_weights)
        
        # 验证
        val_metrics = evaluate_epoch_bert(model, val_loader, device)
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        # 打印进度
        print(f"  训练: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']*100:.2f}%, F1={train_metrics['f1']*100:.2f}%")
        print(f"  验证: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']*100:.2f}%, F1={val_metrics['f1']*100:.2f}%")
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            
            # 保存模型状态（如果是DataParallel，需要保存module.state_dict）
            if isinstance(model, nn.DataParallel):
                best_model_state = model.module.state_dict().copy()
            else:
                best_model_state = model.state_dict().copy()
            
            best_val_predictions = val_metrics['predictions']
            best_val_labels = val_metrics['labels']
            best_val_probs = val_metrics['probabilities']
            
            print(f"  ✓ 新的最佳模型 (val_acc: {val_metrics['accuracy']*100:.2f}%, val_f1: {val_metrics['f1']*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  验证损失未改善 ({patience_counter}/{config.patience})")
        
        # 早停
        if patience_counter >= config.patience:
            print(f"  ⏰ 早停触发，在第{epoch+1}轮停止训练")
            break
        
        # 打印GPU内存使用情况（如果使用GPU）
        if torch.cuda.is_available():
            print(f"  GPU内存: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
    
    # 保存最佳模型
    model_save_path = f'bert_fold_{fold_idx}_best.pth'
    torch.save(best_model_state, model_save_path)
    print(f"✓ 最佳模型已保存到 {model_save_path}")
    
    # 保存transformers格式的模型
    transformers_save_path = f'bert_fold_{fold_idx}_best_transformers'
    if isinstance(model, nn.DataParallel):
        model.module.save_pretrained(transformers_save_path)
    else:
        model.save_pretrained(transformers_save_path)
    tokenizer.save_pretrained(transformers_save_path)
    print(f"✓ Transformers格式模型已保存到 {transformers_save_path}")
    
    # 绘制训练历史
    plot_training_history(history, fold_idx)
    
    # 绘制混淆矩阵
    plot_confusion_matrix_comprehensive(best_val_labels, best_val_predictions, fold_idx)
    
    return {
        'fold': fold_idx,
        'model_config': {
            'model_path': config.model_path,
            'num_labels': config.num_labels,
            'max_length': config.max_length
        },
        'tokenizer': tokenizer,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'history': history,
        'val_predictions': best_val_predictions,
        'val_labels': best_val_labels,
        'val_probabilities': best_val_probs,
        'model_path': model_save_path,
        'transformers_path': transformers_save_path,
        'best_epoch': best_epoch
    }

# ==================== 7. 加载交叉验证数据 ====================
def load_cross_validation_data(data_dir="data"):
    """加载5折交叉验证数据"""
    print("加载交叉验证数据...")
    
    # 检查是否有预分好的折文件
    pre_split_files_exist = True
    for fold_idx in range(5):
        train_path = os.path.join(data_dir, f"train_fold_{fold_idx}.csv")
        val_path = os.path.join(data_dir, f"val_fold_{fold_idx}.csv")
        if not (os.path.exists(train_path) and os.path.exists(val_path)):
            pre_split_files_exist = False
            break
    
    folds = []
    
    if pre_split_files_exist:
        print("📁 使用预分好的交叉验证数据")
        for fold_idx in range(5):
            train_path = os.path.join(data_dir, f"train_fold_{fold_idx}.csv")
            val_path = os.path.join(data_dir, f"val_fold_{fold_idx}.csv")
            
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            
            # 清洗数据
            train_df = clean_data(train_df)
            val_df = clean_data(val_df)
            
            folds.append({
                'fold': fold_idx,
                'train': train_df,
                'val': val_df
            })
            
            print(f"  第{fold_idx+1}折: 训练集={len(train_df)}, 验证集={len(val_df)}")
    else:
        print("📁 从 train.csv 创建交叉验证折")
        # 加载主训练数据
        train_path = os.path.join(data_dir, "train.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到训练数据文件: {train_path}")
        
        train_df = pd.read_csv(train_path)
        print(f"  训练集总大小: {len(train_df)} 行")
        
        # 清洗数据
        train_df = clean_data(train_df)
        print(f"  有效数据: {len(train_df)} 行")
        
        # 创建交叉验证折
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            train_fold = train_df.iloc[train_idx].copy()
            val_fold = train_df.iloc[val_idx].copy()
            
            print(f"  第{fold_idx}折: 训练集 {len(train_fold)} 行, 验证集 {len(val_fold)} 行")
            
            folds.append({
                'fold': fold_idx,
                'train': train_fold,
                'val': val_fold
            })
    
    return folds

# ==================== 8. 主训练流程（5折交叉验证） ====================
def main_bert_cross_validation():
    """BERT模型5折交叉验证训练"""
    print("=" * 70)
    print("🤖 BERT情感分析模型 - 5折交叉验证训练")
    print(f"📱 使用设备: {config.device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"🎮 GPU数量: {torch.cuda.device_count()}")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n" + "="*50)
    print("📂 1. 加载数据")
    print("="*50)
    
    folds_data = load_cross_validation_data()
    
    # 加载测试数据
    test_data_path = os.path.join(config.data_dir, "dev.csv")
    if os.path.exists(test_data_path):
        test_df = pd.read_csv(test_data_path)
        test_df = clean_data(test_df)
        print(f"测试集大小: {len(test_df)}")
    else:
        print("警告: 未找到测试数据文件，将使用验证集进行评估")
        test_df = folds_data[0]['val']  # 使用第一折的验证集作为测试集
    
    # 2. 加载tokenizer
    print("\n" + "="*50)
    print("🔤 2. 加载Tokenizer")
    print("="*50)
    
    print(f"加载BERT tokenizer: {config.model_path}")
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    
    # 3. 创建测试数据集
    print("\n" + "="*50)
    print("🧪 3. 创建测试数据集")
    print("="*50)
    
    test_dataset = BERTDataset(test_df['sentence'].values, test_df['label'].values, tokenizer, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 4. 训练所有折
    print("\n" + "="*50)
    print("🏋️ 4. 训练所有折")
    print("="*50)
    
    all_model_states = []
    all_model_configs = []
    all_fold_results = []
    
    for fold_idx, fold_data in enumerate(folds_data):
        try:
            # 训练单折
            fold_result = train_single_fold_bert(
                fold_idx, 
                fold_data['train'], 
                fold_data['val'], 
                tokenizer, 
                config.device,
                n_epochs=config.num_epochs
            )
            
            # 收集结果
            all_fold_results.append(fold_result)
            
            # 加载最佳模型状态
            best_state = torch.load(fold_result['model_path'], map_location='cpu')
            all_model_states.append(best_state)
            all_model_configs.append(fold_result['model_config'])
            
            print(f"✓ 第{fold_idx+1}折训练完成")
            
        except Exception as e:
            print(f"❌ 第{fold_idx+1}折训练失败: {e}")
            continue
    
    # 5. 评估单个模型和集成模型
    print("\n" + "="*70)
    print("📈 5. 评估模型性能")
    print("="*70)
    
    if not all_model_states:
        raise ValueError("没有成功训练的模型，无法进行评估")
    
    # 预先加载所有模型到GPU
    print("\n预先加载所有模型到内存...")
    loaded_models = []
    for i, (state_dict, model_config) in enumerate(zip(all_model_states, all_model_configs)):
        print(f"  加载第{i+1}个模型...")
        model = BertForSequenceClassification.from_pretrained(
            model_config['model_path'],
            num_labels=model_config['num_labels']
        )
        model.load_state_dict(state_dict)
        model.to(config.device)
        model.eval()
        loaded_models.append(model)
    
    print("✓ 所有模型加载完成")
    
    # 评估单个模型（第一折）
    print("\n评估单个模型（第一折）...")
    single_model = loaded_models[0]
    single_test_metrics = evaluate_epoch_bert(single_model, test_loader, config.device)
    single_test_acc = single_test_metrics['accuracy']
    single_test_f1 = single_test_metrics['f1']
    
    print(f"单模型测试准确率: {single_test_acc*100:.2f}%")
    print(f"单模型测试F1-score: {single_test_f1*100:.2f}%")
    
    # 评估集成模型（软投票）
    print("\n评估集成模型（软投票）...")
    soft_voting_acc = 0
    total_samples = 0
    all_soft_predictions = []
    all_soft_probs = []
    all_test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="集成评估（软投票）"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].cpu().numpy()
            
            # 集成预测（软投票）- 使用预先加载的模型
            all_probs = []
            for model in loaded_models:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.append(probs)
            
            # 平均概率
            avg_probs = torch.mean(torch.stack(all_probs), dim=0)
            predictions = torch.argmax(avg_probs, dim=1).cpu().numpy()
            
            # 收集结果
            all_soft_predictions.extend(predictions)
            all_soft_probs.extend(avg_probs.cpu().numpy())
            all_test_labels.extend(labels)
            
            # 计算准确率
            correct = (predictions == labels).sum()
            soft_voting_acc += correct
            total_samples += len(labels)
    
    ensemble_soft_acc = soft_voting_acc / total_samples
    ensemble_soft_f1 = f1_score(all_test_labels, all_soft_predictions, average='macro')
    print(f"集成模型（软投票）测试准确率: {ensemble_soft_acc*100:.2f}%")
    print(f"集成模型（软投票）测试F1-score: {ensemble_soft_f1*100:.2f}%")
    
    # 评估集成模型（硬投票）
    print("\n评估集成模型（硬投票）...")
    hard_voting_acc = 0
    total_samples = 0
    all_hard_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="集成评估（硬投票）"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].cpu().numpy()
            
            # 收集所有模型的预测 - 使用预先加载的模型
            all_predictions = []
            for model in loaded_models:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_predictions.append(predictions)
            
            # 硬投票：多数票
            all_predictions = np.array(all_predictions)
            final_predictions = []
            
            for i in range(all_predictions.shape[1]):
                votes = all_predictions[:, i]
                vote_0 = np.sum(votes == 0)
                vote_1 = np.sum(votes == 1)
                final_predictions.append(0 if vote_0 > vote_1 else 1)
            
            final_predictions = np.array(final_predictions)
            
            # 收集结果
            all_hard_predictions.extend(final_predictions)
            
            # 计算准确率
            correct = (final_predictions == labels).sum()
            hard_voting_acc += correct
            total_samples += len(labels)
    
    ensemble_hard_acc = hard_voting_acc / total_samples
    ensemble_hard_f1 = f1_score(all_test_labels, all_hard_predictions, average='macro')
    print(f"集成模型（硬投票）测试准确率: {ensemble_hard_acc*100:.2f}%")
    print(f"集成模型（硬投票）测试F1-score: {ensemble_hard_f1*100:.2f}%")
    
    # 6. 生成集成模型可视化
    print("\n" + "="*70)
    print("📈 6. 生成集成模型可视化")
    print("="*70)
    
    # 转换列表为numpy数组
    all_test_labels = np.array(all_test_labels)
    all_soft_predictions = np.array(all_soft_predictions)
    all_soft_probs = np.array(all_soft_probs)
    all_hard_predictions = np.array(all_hard_predictions)
    
    # 绘制集成模型混淆矩阵
    plot_confusion_matrix_comprehensive(all_test_labels, all_soft_predictions, 
                                      model_name='ensemble_soft')
    plot_confusion_matrix_comprehensive(all_test_labels, all_hard_predictions,
                                      model_name='ensemble_hard')
    
    # 7. 保存真正的集成模型
    print("\n" + "="*70)
    print("💾 7. 保存真正的集成模型")
    print("="*70)
    
    true_ensemble_data = {
        'models': all_model_states,
        'model_configs': all_model_configs,
        'tokenizer_info': tokenizer.name_or_path,
        
        'performance': {
            'single_model_acc': float(single_test_acc),
            'single_model_f1': float(single_test_f1),
            'ensemble_soft_acc': float(ensemble_soft_acc),
            'ensemble_soft_f1': float(ensemble_soft_f1),
            'ensemble_hard_acc': float(ensemble_hard_acc),
            'ensemble_hard_f1': float(ensemble_hard_f1),
            'improvement_soft_acc': float(ensemble_soft_acc - single_test_acc),
            'improvement_soft_f1': float(ensemble_soft_f1 - single_test_f1),
            'improvement_hard_acc': float(ensemble_hard_acc - single_test_acc),
            'improvement_hard_f1': float(ensemble_hard_f1 - single_test_f1),
        },
        
        'fold_results': all_fold_results,
        'model_class': 'BertForSequenceClassification',
        'tokenizer_class': 'BertTokenizer',
        'max_len': config.max_length,
        
        'version': '2.0-cross-validation',
        'created_date': time.strftime('%Y-%m-%d'),
        'device': str(config.device),
        'description': 'BERT情感分析 - 5折交叉验证集成模型'
    }
    
    torch.save(true_ensemble_data, 'bert_true_ensemble_model_cv.pt')
    print(f"✓ 真正的集成模型已保存到 bert_true_ensemble_model_cv.pt")
    
    # 同时保存一个transformers格式的集成模型（使用第一折作为代表）
    single_model.save_pretrained('./bert_ensemble_representative')
    tokenizer.save_pretrained('./bert_ensemble_representative')
    
    # 8. 总结
    print("\n" + "="*70)
    print("🎉 BERT模型训练完成总结")
    print("="*70)
    
    print(f"\n📊 最终性能:")
    print(f"  单模型: 准确率={single_test_acc*100:.2f}%, F1={single_test_f1*100:.2f}%")
    print(f"  集成软投票: 准确率={ensemble_soft_acc*100:.2f}%, F1={ensemble_soft_f1*100:.2f}%")
    print(f"  集成硬投票: 准确率={ensemble_hard_acc*100:.2f}%, F1={ensemble_hard_f1*100:.2f}%")
    
    soft_improvement = ensemble_soft_acc - single_test_acc
    hard_improvement = ensemble_hard_acc - single_test_acc
    
    print(f"\n📈 性能提升:")
    print(f"  软投票提升: 准确率 +{soft_improvement*100:.2f}%, F1 +{(ensemble_soft_f1-single_test_f1)*100:.2f}%")
    print(f"  硬投票提升: 准确率 +{hard_improvement*100:.2f}%, F1 +{(ensemble_hard_f1-single_test_f1)*100:.2f}%")
    
    print(f"\n💾 生成的文件:")
    print(f"  1. 单折模型: bert_fold_*_best.pth (5个)")
    print(f"  2. 单折模型(transformers): bert_fold_*_best_transformers/ (5个)")
    print(f"  3. 集成模型: bert_true_ensemble_model_cv.pt")
    print(f"  4. 代表模型: bert_ensemble_representative/")
    print(f"  5. 可视化文件: {config.viz_dir}/ 目录下的所有图片")
    
    return true_ensemble_data

# ==================== 9. 预测函数 ====================
def predict_with_bert_ensemble(text, ensemble_path='bert_true_ensemble_model_cv.pt', device=None):
    """使用集成BERT模型进行预测"""
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载集成模型
    ensemble_data = torch.load(ensemble_path, map_location='cpu', weights_only=False)
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(ensemble_data['tokenizer_info'])
    
    # 处理输入文本
    encoding = tokenizer.encode_plus(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 集成预测（软投票）
    all_probs = []
    for i, (state_dict, model_config) in enumerate(zip(ensemble_data['models'], 
                                                      ensemble_data['model_configs'])):
        model = BertForSequenceClassification.from_pretrained(
            model_config['model_path'],
            num_labels=model_config['num_labels']
        )
        model.load_state_dict(state_dict)
        model.to(device)  # 移动到设备
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs)
    
    # 平均概率
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    prediction = torch.argmax(avg_probs, dim=1).item()
    confidence = torch.max(avg_probs).item()
    
    sentiment = "正面" if prediction == 1 else "负面"
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'prediction': prediction,
        'probabilities': avg_probs.cpu().numpy().tolist()[0],  # 移回CPU
        'device': str(device)
    }

# ==================== 10. 运行 ====================
if __name__ == "__main__":
    try:
        # 运行5折交叉验证训练
        ensemble_data = main_bert_cross_validation()
        
        # 示例：使用集成模型进行预测
        print("\n" + "="*70)
        print("🤖 示例预测")
        print("="*70)
        
        test_texts = [
            "这个商品质量真的很好，非常满意！",
            "物流太慢了，等了整整一个星期",
            "性价比很高，推荐购买",
            "包装破损，商品有瑕疵"
        ]
        
        for text in test_texts:
            result = predict_with_bert_ensemble(text)
            print(f"\n文本: {text}")
            print(f"情感: {result['sentiment']} (置信度: {result['confidence']:.2%})")
            
    except Exception as e:
        print(f"\n程序错误: {e}")
        import traceback
        traceback.print_exc()
