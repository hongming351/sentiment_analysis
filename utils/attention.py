import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """自定义自注意力机制"""
    
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 查询、键、值变换矩阵
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # 缩放因子
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
        # 输出变换
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape
        
        # 计算Q, K, V
        Q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key(x)    # [batch_size, seq_len, hidden_dim]
        V = self.value(x)  # [batch_size, seq_len, hidden_dim]
        
        # 计算注意力分数
        energy = torch.matmul(Q, K.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        
        if self.scale.device != energy.device:
            self.scale = self.scale.to(energy.device)
        
        energy = energy / self.scale
        
        # 应用softmax获取注意力权重
        attention = F.softmax(energy, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # 应用注意力权重到V
        out = torch.matmul(attention, V)  # [batch_size, seq_len, hidden_dim]
        
        # 变换输出维度
        out = self.fc_out(out)
        
        # 全局池化获取序列表示
        out_pooled = out.mean(dim=1)  # [batch_size, hidden_dim]
        
        return out_pooled, attention

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 线性变换层
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 线性变换并分割成多头
        Q = self.fc_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.fc_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.fc_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力
        x = torch.matmul(attention, V)
        
        # 合并多头
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # 输出变换
        x = self.fc_o(x)
        
        # 全局池化
        x_pooled = x.mean(dim=1)
        
        return x_pooled, attention

class BahdanauAttention(nn.Module):
    """Bahdanau注意力机制（加性注意力）"""
    
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        
    def forward(self, query, keys):
        # query: [batch_size, hidden_dim]
        # keys: [batch_size, seq_len, hidden_dim]
        
        # 扩展query维度以匹配keys
        query = query.unsqueeze(1).repeat(1, keys.size(1), 1)  # [batch_size, seq_len, hidden_dim]
        
        # 计算注意力分数
        energy = torch.tanh(self.W(query) + self.U(keys))  # [batch_size, seq_len, hidden_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]
        
        # 应用softmax
        attention_weights = F.softmax(attention, dim=1)  # [batch_size, seq_len]
        
        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)  # [batch_size, hidden_dim]
        
        return context, attention_weights