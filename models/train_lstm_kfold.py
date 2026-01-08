# train_jd_lstm_cpu.py
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import re
import jieba
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter, defaultdict
import collections
import glob
from torch.utils.data import Dataset
print("=" * 70)
print("京东评论情感分析 - LSTM模型训练 (使用交叉验证 - CPU训练)")
print("=" * 70)
# ==================== 1. 设置随机种子 ====================
def set_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU训练，不设置CUDA相关种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1234)

# ==================== 2. 检查并安装jieba ====================
try:
    import jieba
    print("✓ jieba 已安装")
except ImportError:
    print("正在安装 jieba...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba"])
    import jieba
    print("✓ jieba 安装完成")

# ==================== 3. 文本预处理函数 ====================
def clean_text(text):
    """清洗文本"""
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 移除邮箱
    text = re.sub(r'\S+@\S+', '', text)
    # 保留中文、英文、数字和基本标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：,.!?;\'"、]', ' ', text)
    # 合并多个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
def tokenize_chinese(text, use_jieba=True):
    """中文分词"""
    text = clean_text(text)
    if not text:
        return []
    
    if use_jieba:
        # 使用jieba分词
        tokens = jieba.lcut(text)
    else:
        # 简单按字符分割（备用方案）
        tokens = list(text)
    
    # 过滤空字符串
    tokens = [token.strip() for token in tokens if token.strip()]
    
    return tokens

# ==================== 4. 自定义词汇表类 (简化版) ====================
class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # 特殊token（参考IMDB代码，只有unk和pad）
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        
        # 添加特殊token
        self.add_word(self.unk_token)
        self.add_word(self.pad_token)
    
    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def build_vocab(self, texts, tokenizer_fn):
        """从文本构建词汇表"""
        print("构建词汇表中...")
        
        # 统计所有词频
        word_freq = Counter()
        for text in tqdm(texts, desc="处理文本"):
            tokens = tokenizer_fn(text)
            word_freq.update(tokens)
        
        # 过滤低频词
        for word, freq in word_freq.items():
            if freq >= self.min_freq:
                self.add_word(word)
        
        print(f"词汇表构建完成，大小: {len(self)}")
        print(f"  特殊token: 2个 (<unk>, <pad>)")
        print(f"  普通词汇: {len(self) - 2} 个")
        print(f"  最低词频: {self.min_freq}")
    
    def __len__(self):
        return len(self.word2idx)
    
    def word_to_index(self, word):
        """获取词的索引，不存在则返回UNK索引"""
        return self.word2idx.get(word, self.word2idx[self.unk_token])
    
    def index_to_word(self, idx):
        """获取索引对应的词"""
        return self.idx2word.get(idx, self.unk_token)
    
    def lookup_indices(self, tokens):
        """将token列表转换为索引列表（参考IMDB代码中的vocab.lookup_indices）"""
        indices = []
        for token in tokens:
            indices.append(self.word_to_index(token))
        return indices
    
    def set_default_index(self, idx):
        """设置默认索引（为了兼容IMDB代码格式）"""
        # 在这个简化实现中，我们已经在word_to_index中处理了UNK
        pass

# ==================== 5. 数据集类 ====================
class JDDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=128, tokenizer_fn=None):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
        if tokenizer_fn is None:
            self.tokenizer_fn = tokenize_chinese
        else:
            self.tokenizer_fn = tokenizer_fn
        
        # 预处理所有文本
        print("预处理数据中...")
        self.tokens_list = []
        self.lengths = []
        self.ids_list = []
        self.filtered_texts = []
        self.filtered_labels = []
        
        for text, label in tqdm(zip(texts, labels), desc="处理文本", total=len(texts)):
            tokens = self.tokenizer_fn(text)[:max_length]
            length = len(tokens)
            
            # 跳过长度为0的样本
            if length == 0:
                continue
                
            ids = self.vocab.lookup_indices(tokens)
            
            self.tokens_list.append(tokens)
            self.lengths.append(length)
            self.ids_list.append(ids)
            self.filtered_texts.append(text)
            self.filtered_labels.append(label)
        
        # 更新过滤后的数据
        self.texts = self.filtered_texts
        self.labels = self.filtered_labels
        
        print(f"  原始样本数: {len(texts)}")
        print(f"  过滤后样本数: {len(self.texts)} (移除了{len(texts)-len(self.texts)}个空文本)")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'ids': torch.tensor(self.ids_list[idx], dtype=torch.long),
            'length': torch.tensor(self.lengths[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'tokens': self.tokens_list[idx]
        }

# ==================== 6. LSTM模型 (参考IMDB代码结构) ====================
class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        # ids = [batch size, seq len]
        # length = [batch size]
        embedded = self.dropout(self.embedding(ids))
        # embedded = [batch size, seq len, embedding dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [batch size, seq len, hidden dim * n directions]
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = self.dropout(hidden[-1])
            # hidden = [batch size, hidden dim]
        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]
        return prediction

# ==================== 7. 数据加载器函数 ====================
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_length = [i["length"] for i in batch]
        batch_length = torch.stack(batch_length)
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "length": batch_length, "label": batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

# ==================== 8. 训练和评估函数 (参考IMDB代码) ====================
def train(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm(dataloader, desc="training..."):
        ids = batch["ids"].to(device)
        length = batch["length"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            length = batch["length"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param)

# ==================== 9. 预测函数 ====================
def predict_sentiment(text, model, tokenizer_fn, vocab, device, max_length=256):
    tokens = tokenizer_fn(text)[:max_length]
    ids = vocab.lookup_indices(tokens)
    length = torch.LongTensor([len(ids)])
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor, length).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability, probability[0].item(), probability[1].item()

# ==================== 10. 数据加载函数 ====================
def load_cross_validation_data(data_dir):
    """加载交叉验证数据集"""
    print(f"正在加载交叉验证数据，目录: {data_dir}")

    # 查找所有训练集和验证集文件
    train_files = sorted(glob.glob(os.path.join(data_dir, "train_fold_*.csv")))
    valid_files = sorted(glob.glob(os.path.join(data_dir, "val_fold_*.csv")))

    if train_files and valid_files:
        # 如果找到交叉验证文件，使用它们
        print(f"找到 {len(train_files)} 个训练集文件和 {len(valid_files)} 个验证集文件")

        # 合并所有训练集和验证集数据
        all_train_dfs = []
        all_valid_dfs = []

        for train_file in train_files:
            df = pd.read_csv(train_file)
            # 处理缺失值
            df = df.dropna(subset=['sentence', 'label'])
            df['label'] = df['label'].fillna(0).astype(int)
            all_train_dfs.append(df)
            print(f"  加载: {os.path.basename(train_file)} - {len(df)} 条数据")

        for valid_file in valid_files:
            df = pd.read_csv(valid_file)
            # 处理缺失值
            df = df.dropna(subset=['sentence', 'label'])
            df['label'] = df['label'].fillna(0).astype(int)
            all_valid_dfs.append(df)
            print(f"  加载: {os.path.basename(valid_file)} - {len(df)} 条数据")

        # 合并数据
        train_df = pd.concat(all_train_dfs, ignore_index=True)
        valid_df = pd.concat(all_valid_dfs, ignore_index=True)

        print(f"✓ 合并后训练集: {len(train_df)} 条评论")
        print(f"✓ 合并后验证集: {len(valid_df)} 条评论")
    else:
        # 如果没有找到交叉验证文件，使用train.csv和dev.csv作为替代
        print("⚠️  未找到交叉验证文件，使用train.csv和dev.csv作为替代")

        # 加载train.csv作为训练集
        train_file = os.path.join(data_dir, "train.csv")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"在目录 {data_dir} 中找不到train.csv文件")

        train_df = pd.read_csv(train_file)
        # 处理缺失值
        train_df = train_df.dropna(subset=['sentence', 'label'])
        train_df['label'] = train_df['label'].fillna(0).astype(int)
        print(f"  加载训练集: train.csv - {len(train_df)} 条数据")

        # 加载dev.csv作为验证集
        valid_file = os.path.join(data_dir, "dev.csv")
        if not os.path.exists(valid_file):
            raise FileNotFoundError(f"在目录 {data_dir} 中找不到dev.csv文件")

        valid_df = pd.read_csv(valid_file)
        # 处理缺失值
        valid_df = valid_df.dropna(subset=['sentence', 'label'])
        valid_df['label'] = valid_df['label'].fillna(0).astype(int)
        print(f"  加载验证集: dev.csv - {len(valid_df)} 条数据")

        print(f"✓ 使用train.csv作为训练集: {len(train_df)} 条评论")
        print(f"✓ 使用dev.csv作为验证集: {len(valid_df)} 条评论")

    return train_df, valid_df

def load_test_data(test_file, data_dir=None):
    """加载测试数据"""
    print(f"正在加载测试数据: {test_file}")
    
    # 如果文件不存在，尝试在data_dir中查找
    if not os.path.exists(test_file) and data_dir:
        test_file_in_data_dir = os.path.join(data_dir, os.path.basename(test_file))
        if os.path.exists(test_file_in_data_dir):
            test_file = test_file_in_data_dir
            print(f"  在数据目录中找到: {test_file}")
        else:
            raise FileNotFoundError(f"找不到测试文件: {test_file}")
    
    test_df = pd.read_csv(test_file)
    
    # 处理缺失值
    test_df = test_df.dropna(subset=['sentence', 'label'])  # 删除sentence或label为NaN的行
    test_df['label'] = test_df['label'].fillna(0).astype(int)  # 填充剩余的NaN为0
    
    print(f"✓ 测试集: {len(test_df)} 条评论 (清理后)")
    
    return test_df

# ==================== 11. 主函数 ====================
def main():
    # ==================== 设备选择 ====================
    print("\n" + "="*70)
    print("🔧 设备选择与配置")
    print("="*70)

    # 优先使用GPU，如果不可用则使用CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"📱 使用设备: {device}")
        print(f"  ✅ GPU可用 - 使用GPU加速训练")
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print(f"📱 使用设备: {device}")
        print(f"  ⚠️  CUDA不可用，使用CPU")
        print(f"  训练速度可能较慢，请耐心等待")
    
    # ==================== 数据目录配置 ====================
    # 使用当前项目的数据目录
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    test_file = os.path.join(data_dir, "dev.csv")
    
    # ==================== 加载数据 ====================
    print("\n" + "="*70)
    print("📊 加载数据")
    print("="*70)
    
    try:
        # 加载交叉验证数据作为训练集和验证集
        train_df, valid_df = load_cross_validation_data(data_dir)
        
        # 加载dev.csv作为测试集
        test_df = load_test_data(test_file)
        
        # 检查数据格式
        print(f"\n📋 数据列名:")
        print(f"  训练集: {list(train_df.columns)}")
        print(f"  验证集: {list(valid_df.columns)}")
        print(f"  测试集: {list(test_df.columns)}")
        
        # 标签分布
        train_labels = train_df['label'].value_counts().sort_index()
        valid_labels = valid_df['label'].value_counts().sort_index()
        test_labels = test_df['label'].value_counts().sort_index()
        
        print(f"\n🎯 标签分布:")
        for label in [0, 1]:
            train_count = train_labels.get(label, 0)
            valid_count = valid_labels.get(label, 0)
            test_count = test_labels.get(label, 0)
            label_name = "负面" if label == 0 else "正面"
            print(f"  {label_name} (标签={label}):")
            print(f"    训练集: {train_count} 条 ({train_count/len(train_df)*100:.1f}%)")
            print(f"    验证集: {valid_count} 条 ({valid_count/len(valid_df)*100:.1f}%)")
            print(f"    测试集: {test_count} 条 ({test_count/len(test_df)*100:.1f}%)")
        
        # 显示示例
        print(f"\n🔍 数据示例:")
        for i in range(min(2, len(train_df))):
            text = train_df.iloc[i]['sentence']
            label = train_df.iloc[i]['label']
            sentiment = "负面" if label == 0 else "正面"
            print(f"  训练集示例 {i+1}:")
            print(f"    文本: {text[:60]}...")
            print(f"    标签: {label} ({sentiment})")
            print(f"    分词: {tokenize_chinese(text)[:12]}...")
            
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print(f"请确保数据文件存在")
        print(f"数据目录: {data_dir}")
        print(f"测试文件: {test_file}")
        if os.path.exists(data_dir):
            print(f"数据目录内容: {os.listdir(data_dir)}")
        return
    
    # ==================== 构建词汇表 ====================
    print("\n" + "="*70)
    print("📚 构建词汇表")
    print("="*70)
    
    # 创建词汇表
    min_freq = 2
    vocab = Vocabulary(min_freq=min_freq)
    
    # 从训练数据构建词汇表（只使用训练数据）
    train_texts = train_df['sentence'].astype(str).tolist()
    vocab.build_vocab(train_texts, tokenize_chinese)
    
    # 设置特殊token的索引
    unk_index = vocab.word2idx[vocab.unk_token]
    pad_index = vocab.word2idx[vocab.pad_token]
    vocab.set_default_index(unk_index)
    
    print(f"  UNK索引: {unk_index}")
    print(f"  PAD索引: {pad_index}")
    
    # ==================== 创建数据集 ====================
    print("\n" + "="*70)
    print("📁 创建数据集")
    print("="*70)
    
    # 数据集参数（使用较小参数以加快CPU训练速度）
    max_length = 128
    
    # 创建训练集、验证集和测试集
    train_dataset = JDDataset(
        train_df['sentence'].astype(str).tolist(),
        train_df['label'].astype(int).tolist(),
        vocab,
        max_length=max_length,
        tokenizer_fn=tokenize_chinese
    )
    
    valid_dataset = JDDataset(
        valid_df['sentence'].astype(str).tolist(),
        valid_df['label'].astype(int).tolist(),
        vocab,
        max_length=max_length,
        tokenizer_fn=tokenize_chinese
    )
    
    test_dataset = JDDataset(
        test_df['sentence'].astype(str).tolist(),
        test_df['label'].astype(int).tolist(),
        vocab,
        max_length=max_length,
        tokenizer_fn=tokenize_chinese
    )
    
    # ==================== 创建数据加载器 ====================
    # 使用较小的批处理大小以适应CPU内存
    batch_size = 16
    
    train_data_loader = get_data_loader(train_dataset, batch_size, pad_index, shuffle=True)
    valid_data_loader = get_data_loader(valid_dataset, batch_size, pad_index)
    test_data_loader = get_data_loader(test_dataset, batch_size, pad_index)
    
    print(f"✓ 最大序列长度: {max_length}")
    print(f"✓ 批处理大小: {batch_size} (适应CPU内存)")
    print(f"✓ 训练批次数: {len(train_data_loader)}")
    print(f"✓ 验证批次数: {len(valid_data_loader)}")
    print(f"✓ 测试批次数: {len(test_data_loader)}")
    
    # ==================== 创建模型 ====================
    print("\n" + "="*70)
    print("🧠 创建LSTM模型 (CPU优化版本)")
    print("="*70)
    
    # 模型参数（使用较小参数以加快CPU训练速度）
    vocab_size = len(vocab)
    embedding_dim = 100  # 减小嵌入维度
    hidden_dim = 100     # 减小隐藏层维度
    output_dim = 2
    n_layers = 2
    bidirectional = True
    dropout_rate = 0.3   # 降低dropout
    
    # 创建模型
    model = LSTM(
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    )
    
    # 初始化权重
    model.apply(initialize_weights)
    
    # 将模型移到CPU
    model = model.to(device)
    
    print(f"✓ 词汇表大小: {vocab_size}")
    print(f"✓ 嵌入维度: {embedding_dim} (CPU优化)")
    print(f"✓ 隐藏层维度: {hidden_dim} (CPU优化)")
    print(f"✓ LSTM层数: {n_layers}")
    print(f"✓ 双向: {bidirectional}")
    print(f"✓ Dropout: {dropout_rate}")
    print(f"✓ 模型参数量: {count_parameters(model):,}")
    print(f"✓ 模型已加载到: {device}")
    
    # ==================== 训练配置 ====================
    print("\n" + "="*70)
    print("⚙️ 训练配置")
    print("="*70)
    
    # 优化器（使用较小学习率）
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    print(f"✓ 优化器: Adam (lr={lr})")
    print(f"✓ 损失函数: CrossEntropyLoss")
    print(f"✓ 所有计算都在CPU上进行")
    
    # ==================== 训练循环 ====================
    print("\n" + "="*70)
    print("🚀 开始训练 (CPU训练，请耐心等待...)")
    print("="*70)
    
    n_epochs = 15  # 增加epoch数，因为CPU训练较慢但稳定
    best_valid_loss = float("inf")
    
    # 记录训练历史
    metrics = collections.defaultdict(list)
    
    for epoch in range(n_epochs):
        print(f"\n📈 Epoch {epoch+1}/{n_epochs}")
        
        # 训练
        train_loss, train_acc = train(
            train_data_loader, model, criterion, optimizer, device
        )
        
        # 评估
        valid_loss, valid_acc = evaluate(
            valid_data_loader, model, criterion, device
        )
        
        # 记录历史
        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)
        
        # 打印结果
        print(f"epoch: {epoch+1}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "jd_lstm_best_cpu.pt")
            print(f"✓ 保存最佳模型到 jd_lstm_best_cpu.pt")
        
        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"jd_lstm_epoch_{epoch+1}_cpu.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, checkpoint_path)
            print(f"✓ 保存检查点到 {checkpoint_path}")
    
    # ==================== 可视化结果 ====================
    print("\n" + "="*70)
    print("📊 训练结果可视化")
    print("="*70)
    
    # 创建可视化目录
    os.makedirs("results_cpu", exist_ok=True)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["train_losses"], label="train loss", marker='o')
    ax.plot(metrics["valid_losses"], label="valid loss", marker='s')
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()
    plt.title("训练和验证损失 (CPU训练)")
    plt.savefig("results_cpu/training_loss_cpu.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["train_accs"], label="train accuracy", marker='o')
    ax.plot(metrics["valid_accs"], label="valid accuracy", marker='s')
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()
    plt.title("训练和验证准确率 (CPU训练)")
    plt.savefig("results_cpu/training_accuracy_cpu.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ 训练曲线已保存到 results_cpu/training_loss_cpu.png 和 results_cpu/training_accuracy_cpu.png")
    
    # ==================== 加载最佳模型并在测试集上评估 ====================
    print("\n" + "="*70)
    print("🧪 在测试集上评估模型")
    print("="*70)
    
    # 加载最佳模型
    model.load_state_dict(torch.load("jd_lstm_best_cpu.pt"))
    model.eval()
    
    # 在测试集上评估
    test_loss, test_acc = evaluate(
        test_data_loader, model, criterion, device
    )
    
    print(f"📊 测试集结果:")
    print(f"  测试损失: {test_loss:.4f}")
    print(f"  测试准确率: {test_acc:.4f}")
    print(f"  测试准确率百分比: {test_acc*100:.2f}%")
    
    # ==================== 测试示例 ====================
    print("\n" + "="*70)
    print("🔍 示例预测")
    print("="*70)
    
    # 从测试集中随机选择一些样本
    np.random.seed(42)
    sample_indices = np.random.choice(len(test_df), 8, replace=False)
    
    print("测试集示例预测:")
    print("-" * 70)
    for idx in sample_indices:
        text = test_df.iloc[idx]['sentence']
        true_label = test_df.iloc[idx]['label']
        true_sentiment = "正面" if true_label == 1 else "负面"
        
        predicted_class, confidence, neg_prob, pos_prob = predict_sentiment(
            text, model, tokenize_chinese, vocab, device, max_length
        )
        predicted_sentiment = "正面" if predicted_class == 1 else "负面"
        
        # 检查预测是否正确
        correct = "✓" if predicted_class == true_label else "✗"
        
        print(f"📝 文本: {text[:80]}...")
        print(f"  真实情感: {true_sentiment} (标签={true_label})")
        print(f"  预测情感: {predicted_sentiment} {correct} (置信度: {confidence:.2%})")
        print(f"  负面概率: {neg_prob:.4f}, 正面概率: {pos_prob:.4f}")
        print()
    
    # 自定义测试文本
    test_texts = [
        "手机质量很好，运行流畅，非常满意！",
        "物流太慢了，等了整整一周才到货",
        "客服态度很差，解决问题效率低",
        "包装很精美，送货速度也很快",
        "商品与描述不符，有质量问题",
        "性价比很高，物超所值",
        "屏幕有划痕，品控需要加强",
        "操作简单，适合老年人使用"
    ]
    
    print("自定义文本预测:")
    print("-" * 70)
    for text in test_texts:
        predicted_class, confidence, neg_prob, pos_prob = predict_sentiment(
            text, model, tokenize_chinese, vocab, device, max_length
        )
        sentiment = "正面" if predicted_class == 1 else "负面"
        print(f"📝 文本: {text}")
        print(f"  预测情感: {sentiment} (置信度: {confidence:.2%})")
        print(f"  负面概率: {neg_prob:.4f}, 正面概率: {pos_prob:.4f}")
        print()
    
    # ==================== 保存结果 ====================
    print("\n" + "="*70)
    print("💾 保存模型和结果")
    print("="*70)
    
    # 保存词汇表
    vocab_data = {
        'word2idx': vocab.word2idx,
        'idx2word': vocab.idx2word,
        'unk_token': vocab.unk_token,
        'pad_token': vocab.pad_token,
        'min_freq': vocab.min_freq
    }
    torch.save(vocab_data, "jd_vocab_cpu.pt")
    
    # 保存完整模型信息
    model_info = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'n_layers': n_layers,
        'bidirectional': bidirectional,
        'dropout_rate': dropout_rate,
        'pad_index': pad_index,
        'max_length': max_length,
        'metrics': metrics,
        'test_results': {'loss': test_loss, 'accuracy': test_acc},
        'device': str(device),
        'training_info': 'CPU训练版本'
    }
    torch.save(model_info, "jd_lstm_full_cpu.pt")
    
    # 保存测试结果到CSV
    test_results = []
    for i in range(len(test_df)):
        text = test_df.iloc[i]['sentence']
        true_label = test_df.iloc[i]['label']
        
        predicted_class, confidence, neg_prob, pos_prob = predict_sentiment(
            text, model, tokenize_chinese, vocab, device, max_length
        )
        
        test_results.append({
            'text': text,
            'true_label': true_label,
            'predicted_label': predicted_class,
            'confidence': confidence,
            'neg_prob': neg_prob,
            'pos_prob': pos_prob,
            'correct': 1 if predicted_class == true_label else 0
        })
    
    results_df = pd.DataFrame(test_results)
    results_df.to_csv("test_predictions_cpu.csv", index=False, encoding='utf-8-sig')
    
    # 计算并保存评估指标
    accuracy = results_df['correct'].mean()
    confusion_matrix = pd.crosstab(
        results_df['true_label'], 
        results_df['predicted_label'],
        rownames=['True'], 
        colnames=['Predicted']
    )
    
    with open("evaluation_results_cpu.txt", "w", encoding='utf-8') as f:
        f.write("京东评论情感分析 - CPU训练评估结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"训练设备: {device}\n")
        f.write(f"测试集大小: {len(test_df)}\n")
        f.write(f"测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"测试损失: {test_loss:.4f}\n\n")
        f.write("模型参数:\n")
        f.write(f"  嵌入维度: {embedding_dim}\n")
        f.write(f"  隐藏层维度: {hidden_dim}\n")
        f.write(f"  LSTM层数: {n_layers}\n")
        f.write(f"  批处理大小: {batch_size}\n")
        f.write(f"  最大序列长度: {max_length}\n\n")
        f.write("混淆矩阵:\n")
        f.write(str(confusion_matrix) + "\n\n")
        f.write("详细结果已保存到 test_predictions_cpu.csv\n")
    
    print(f"✓ 词汇表已保存到 jd_vocab_cpu.pt")
    print(f"✓ 完整模型信息已保存到 jd_lstm_full_cpu.pt")
    print(f"✓ 测试预测结果已保存到 test_predictions_cpu.csv")
    print(f"✓ 评估结果已保存到 evaluation_results_cpu.txt")
    print(f"\n🎉 CPU训练和评估完成！")
    print(f"📁 所有结果文件都保存在当前目录，前缀为 '_cpu'")
    print(f"⏱️  感谢您的耐心等待，CPU训练可能需要一些时间")

# ==================== 运行主函数 ====================
if __name__ == "__main__":
    main()