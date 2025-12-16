
"""
model_source.py
LSTM+Attention 模型源代码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttentionModel(nn.Module):
    """LSTM+Attention 情感分析模型"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, 
                 num_layers=2, dropout=0.3, bidirectional=True):
        super(LSTMAttentionModel, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制： 自动学习哪些词更重要
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 分类头：输出正面/负面情感预测
        self.dropout = nn.Dropout(dropout)
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # 二分类
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len]
            
        Returns:
            logits: 分类logits [batch_size, 2]
            attention_weights: 注意力权重 [batch_size, seq_len]
        """
        # 嵌入层
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM层
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 加权求和
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 分类
        logits = self.classifier(context)
        
        return logits, attention_weights.squeeze(-1)
    
    def get_attention_weights(self, x):
        """
        获取注意力权重（用于可视化）
        
        Args:
            x: 输入张量 [batch_size, seq_len]
            
        Returns:
            attention_weights: 注意力权重 [batch_size, seq_len]
        """
        with torch.no_grad():
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            attention_weights = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)
            return attention_weights.squeeze(-1)
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'vocab_size': self.embedding.num_embeddings,
            'embedding_dim': self.embedding.embedding_dim,
            'hidden_dim': self.lstm.hidden_size,
            'num_layers': self.lstm.num_layers,
            'dropout': self.dropout.p,
            'bidirectional': self.lstm.bidirectional,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class OptimizedLSTM(nn.Module):
    """
    优化版LSTM+Attention模型（兼容旧版本）
    保持与之前训练的模型相同的接口
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super(OptimizedLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        """前向传播（与旧版本兼容）"""
        # 嵌入层
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM层
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 加权求和
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 分类
        output = self.fc(context)
        
        return output, attention_weights.squeeze(-1)


class SimpleLSTM(nn.Module):
    """
    简化版LSTM模型（用于快速实验）
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=1, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        """前向传播"""
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 分类
        output = self.fc(last_output)
        
        return output, None  # 返回None作为占位符，保持接口一致


def create_model(model_type='lstm_attention', **kwargs):
    """
    创建模型工厂函数
    
    Args:
        model_type: 模型类型，可选 'lstm_attention', 'optimized_lstm', 'simple_lstm'
        **kwargs: 传递给模型构造函数的参数
        
    Returns:
        模型实例
    """
    model_classes = {
        'lstm_attention': LSTMAttentionModel,
        'optimized_lstm': OptimizedLSTM,
        'simple_lstm': SimpleLSTM
    }
    
    if model_type not in model_classes:
        raise ValueError(f"未知的模型类型: {model_type}。可选: {list(model_classes.keys())}")
    
    return model_classes[model_type](**kwargs)


def load_model_from_checkpoint(checkpoint_path, model_type='lstm_attention', device=None):
    """
    从检查点加载模型
    
    Args:
        checkpoint_path: 检查点文件路径
        model_type: 模型类型
        device: 设备，如果为None则自动选择
        
    Returns:
        model: 加载的模型
        checkpoint: 检查点数据
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型配置
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # 兼容旧版本
        config = {
            'vocab_size': checkpoint.get('vocab_size', 10000),
            'embedding_dim': 128,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True
        }
    
    # 创建模型
    if model_type == 'lstm_attention':
        model = LSTMAttentionModel(
            vocab_size=config.get('vocab_size'),
            embedding_dim=config.get('embedding_dim', 128),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3),
            bidirectional=config.get('bidirectional', True)
        )
    elif model_type == 'optimized_lstm':
        model = OptimizedLSTM(
            vocab_size=config.get('vocab_size'),
            embedding_dim=config.get('embedding_dim', 128),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3)
        )
    else:
        model = SimpleLSTM(
            vocab_size=config.get('vocab_size'),
            embedding_dim=config.get('embedding_dim', 128),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3)
        )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


# 测试代码
if __name__ == "__main__":
    print("测试模型类...")
    
    # 测试LSTMAttentionModel
    vocab_size = 10000
    model = LSTMAttentionModel(vocab_size=vocab_size)
    
    print(f"模型类型: {model.__class__.__name__}")
    print(f"模型信息: {model.get_model_info()}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, attention_weights = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 测试模型工厂
    print("\n测试模型工厂...")
    for model_type in ['lstm_attention', 'optimized_lstm', 'simple_lstm']:
        test_model = create_model(model_type=model_type, vocab_size=vocab_size)
        print(f"{model_type}: {test_model.__class__.__name__}")
    
    print("\n✅ 所有测试通过!")