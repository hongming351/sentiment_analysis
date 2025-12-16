
"""
数据加载模块
"""

import pandas as pd
from pathlib import Path

class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        
    def load_jd_reviews(self):
        """加载京东评论数据"""
        train_path = self.data_dir / "train.csv"
        dev_path = self.data_dir / "dev.csv"
        # 加载数据
        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)
        
        # 标准化列名
        train_df = self._standardize_columns(train_df)
        dev_df = self._standardize_columns(dev_df)
        
        # 添加数据集标识
        train_df['dataset'] = 'train'
        dev_df['dataset'] = 'dev'
        
        # 合并数据
        combined_df = pd.concat([train_df, dev_df], ignore_index=True)
        
        print(f"数据加载完成: 训练集 {len(train_df)}条, 验证集 {len(dev_df)}条")
        
        return combined_df
    
    def _standardize_columns(self, df):
        """标准化列名"""
        column_mapping = {}
        
        # 检测文本列
        text_cols = ['sentence', 'review', 'text', 'content', 'comment']
        for col in text_cols:
            if col in df.columns:
                column_mapping[col] = 'review'
                break
        
        # 检测标签列
        label_cols = ['label', 'sentiment', 'score', 'rating']
        for col in label_cols:
            if col in df.columns:
                column_mapping[col] = 'sentiment'
                break
        
        # 重命名列
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        return df
    
    def get_train_test_split(self, use_original_split=True):
        """获取训练测试划分"""
        df = self.load_jd_reviews()
        
        if use_original_split and 'dataset' in df.columns:
            # 使用原始划分
            train_df = df[df['dataset'] == 'train']
            test_df = df[df['dataset'] == 'dev']
            
            X_train = train_df['review']
            X_test = test_df['review']
            y_train = train_df['sentiment']
            y_test = test_df['sentiment']
        else:
            # 随机划分
            from sklearn.model_selection import train_test_split
            
            X = df['review']
            y = df['sentiment']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        return X_train, X_test, y_train, y_test
    
    def analyze_dataset(self):
        """分析数据集"""
        df = self.load_jd_reviews()
        
        stats = {
            'total_samples': len(df),
            'columns': list(df.columns),
            'sentiment_distribution': None,
            'text_length_stats': None,
            'dataset_distribution': None
        }
        
        # 情感分布
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            stats['sentiment_distribution'] = {
                'total_positive': int(sentiment_counts.get(1, 0)),
                'total_negative': int(sentiment_counts.get(0, 0)),
                'ratios': (sentiment_counts / len(df)).to_dict()
            }
        
        # 文本长度
        if 'review' in df.columns:
            text_lengths = df['review'].astype(str).str.len()
            stats['text_length_stats'] = {
                'mean': float(text_lengths.mean()),
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'median': float(text_lengths.median())
            }
        
        # 数据集分布
        if 'dataset' in df.columns:
            dataset_counts = df['dataset'].value_counts()
            stats['dataset_distribution'] = {
                'counts': dataset_counts.to_dict(),
                'ratios': (dataset_counts / len(df)).to_dict()
            }
        
        return stats