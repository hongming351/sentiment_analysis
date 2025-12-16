
"""
数据预处理模块 - 京东评论情感分析专用版
"""

import pandas as pd
import re
import jieba
from sklearn.utils import resample
import warnings
from pathlib import Path
import pickle
import json
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """京东评论数据预处理类"""
    
    def __init__(self, use_jieba=True, custom_stop_words=None):
        """
        初始化预处理类
        
        Args:
            use_jieba: 是否使用jieba分词
            custom_stop_words: 自定义停用词列表
        """
        self.use_jieba = use_jieba
        
        # 初始化jieba
        try:
            jieba.initialize()
        except:
            pass
        
        # 京东评论特定的停用词（扩展版）
        self.jd_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '买', '东西', '商品', '京东', '这个', '那个', '就是', '但是', '不过',
            '然后', '所以', '感觉', '觉得', '没有', '不是', '真的', '一点', '一些', '就是',
            '非常', '比较', '太', '真', '好', '没有', '什么', '这样', '那样', '怎么', '为什么',
            '哪里', '哪个', '多少', '几', '第', '着', '过', '地', '得', '啊', '呀', '呢',
            '吗', '吧', '哦', '唉', '嗯', '呃', '呵', '哈', '哼', '呸', '哟', '喔', '哇',
            '喂', '嘛', '唉', '啊', '呀', '还', '又', '都', '再', '才', '就', '却', '只',
            '可', '能', '会', '可以', '可能', '应该', '必须', '需要', '想要', '希望',
            '觉得', '感觉', '认为', '以为', '知道', '了解', '明白', '懂', '学习',
            '工作', '生活', '时间', '今天', '明天', '昨天', '现在', '以后', '以前',
            '配送', '快递', '物流', '发货', '包装', '客服', '服务', '质量', '价格',
            '京东', '淘宝', '天猫', '拼多多', '平台', '网站', 'app', '手机', '电脑',
            '购买', '下单', '订单', '收货', '评价', '评论', '评分', '星级', '星',
            '商品', '产品', '物品', '货物', '宝贝', '东西', '用品', '设备', '机器',
            '使用', '试用', '体验', '感受', '效果', '性能', '功能', '特点', '特色'
        }
        
        # 如果有自定义停用词，合并
        if custom_stop_words:
            self.jd_stop_words.update(custom_stop_words)
    
    def clean_jd_text(self, text):
        """
        专门针对京东评论的文本清理
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # 转换为字符串并去空格
        text = str(text).strip()
        
        # 移除HTML实体
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        
        # 移除常见的HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 处理京东评论中的特殊格式
        text = re.sub(r'【.*?】', '', text)  # 移除【】内容
        text = re.sub(r'\[.*?\]', '', text)  # 移除[]内容
        
        # 处理数字评分（如：5分，五星）
        text = re.sub(r'\d+分', ' ', text)
        text = re.sub(r'\d+星', ' ', text)
        
        # 移除特殊字符但保留中文和英文
        text = re.sub(r'[^\w\u4e00-\u9fff\s,.!?;:]', ' ', text)
        
        # 移除重复的标点
        text = re.sub(r'[.!?;:]{2,}', '.', text)
        
        # 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def segment_jd_text(self, text, remove_stopwords=True):
        """
        针对京东评论的分词
        
        Args:
            text: 清理后的文本
            remove_stopwords: 是否移除停用词
            
        Returns:
            分词后的文本
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        text = text.strip()
        
        if self.use_jieba:
            # 使用jieba分词
            words = jieba.lcut(text)
        else:
            # 简单空格分词
            words = text.split()
        
        # 过滤和处理
        processed_words = []
        for word in words:
            word = word.strip()
            
            # 过滤空词
            if not word:
                continue
            
            # 过滤纯数字
            if word.isdigit():
                continue
            
            # 过滤短词（但保留有意义的单字词）
            if len(word) == 1 and word not in {'好', '差', '贵', '值', '快', '慢', '赞'}:
                continue
            
            # 移除停用词
            if remove_stopwords and word in self.jd_stop_words:
                continue
            
            processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def build_vocabulary(self, texts, max_vocab_size=20000, min_freq=2):
        """
        构建词汇表
        
        Args:
            texts: 分词后的文本列表
            max_vocab_size: 最大词汇表大小
            min_freq: 最小词频
            
        Returns:
            vocab: 词汇表字典
            word_freq: 词频统计
        """
        from collections import Counter
        
        # 统计词频
        word_counter = Counter()
        for text in texts:
            if isinstance(text, str):
                words = text.split()
                word_counter.update(words)
        
        print(f"原始词数: {len(word_counter)}")
        
        # 过滤低频词
        filtered_words = {word: freq for word, freq in word_counter.items() 
                         if freq >= min_freq}
        
        print(f"过滤低频词后: {len(filtered_words)} (min_freq={min_freq})")
        
        # 按频率排序
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        
        # 构建词汇表
        vocab = {
            '<PAD>': 0,  # 填充标记
            '<UNK>': 1,  # 未知词标记
            '<BOS>': 2,  # 开始标记
            '<EOS>': 3   # 结束标记
        }
        
        # 添加实际词汇
        vocab_size = min(max_vocab_size - 4, len(sorted_words))
        for idx, (word, freq) in enumerate(sorted_words[:vocab_size]):
            vocab[word] = idx + 4
        
        print(f"最终词汇表大小: {len(vocab)}")
        print(f"Top 10高频词: {sorted_words[:10]}")
        
        return vocab, word_counter
    
    def text_to_indices(self, text, vocab, max_length=100):
        """
        将文本转换为索引序列
        
        Args:
            text: 分词后的文本
            vocab: 词汇表
            max_length: 最大序列长度
            
        Returns:
            索引列表
        """
        if not isinstance(text, str):
            return [vocab['<PAD>']] * max_length
        
        words = text.split()
        indices = []
        
        # 添加开始标记
        indices.append(vocab.get('<BOS>', vocab['<UNK>']))
        
        # 转换词到索引
        for word in words[:max_length-2]:  # 留位置给结束标记
            indices.append(vocab.get(word, vocab['<UNK>']))
        
        # 添加结束标记
        indices.append(vocab.get('<EOS>', vocab['<UNK>']))
        
        # 填充或截断
        if len(indices) < max_length:
            indices += [vocab['<PAD>']] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        
        return indices
    
    def preprocess_jd_dataset(self, df, text_column='sentence', label_column='label', 
                            balance=False, save_vocab_path=None):
        """
        预处理京东评论数据集
        
        Args:
            df: 原始DataFrame
            text_column: 文本列名
            label_column: 标签列名
            balance: 是否平衡数据集
            save_vocab_path: 保存词汇表的路径
            
        Returns:
            处理后的DataFrame和词汇表
        """
        print("="*60)
        print("京东评论数据预处理")
        print("="*60)
        
        # 复制数据
        processed_df = df.copy()
        original_len = len(processed_df)
        
        print(f"原始数据: {original_len:,} 条")
        
        # 1. 清理文本
        print("\n1. 清理文本...")
        processed_df['cleaned_text'] = processed_df[text_column].apply(self.clean_jd_text)
        
        # 移除空文本
        processed_df = processed_df[processed_df['cleaned_text'].str.len() > 0]
        print(f"  清理后: {len(processed_df):,} 条 (移除 {original_len - len(processed_df):,} 条)")
        
        # 2. 分词
        print("\n2. 分词...")
        processed_df['segmented_text'] = processed_df['cleaned_text'].apply(
            lambda x: self.segment_jd_text(x, remove_stopwords=True)
        )
        
        # 移除分词后为空的数据
        processed_df = processed_df[processed_df['segmented_text'].str.len() > 0]
        print(f"  分词后: {len(processed_df):,} 条")
        
        # 3. 处理标签
        if label_column in processed_df.columns:
            print(f"\n3. 处理标签 ({label_column})...")
            
            # 确保标签是数值
            processed_df[label_column] = pd.to_numeric(processed_df[label_column], errors='coerce')
            
            # 移除无效标签
            before_label = len(processed_df)
            processed_df = processed_df.dropna(subset=[label_column])
            after_label = len(processed_df)
            print(f"  有效标签: {after_label:,} 条 (移除 {before_label - after_label:,} 条无效标签)")
            
            # 转换为整数
            processed_df[label_column] = processed_df[label_column].astype(int)
            
            # 统计标签分布
            label_dist = processed_df[label_column].value_counts().sort_index()
            print(f"  标签分布:")
            for label, count in label_dist.items():
                percentage = count / len(processed_df) * 100
                label_name = "正面" if label == 1 else "负面"
                print(f"    {label_name}({label}): {count:,} 条 ({percentage:.1f}%)")
        
        # 4. 平衡数据集（可选）
        if balance and label_column in processed_df.columns:
            print("\n4. 平衡数据集...")
            processed_df = self.balance_jd_dataset(processed_df, label_column)
            print(f"  平衡后: {len(processed_df):,} 条")
        
        # 5. 构建词汇表
        print("\n5. 构建词汇表...")
        texts = processed_df['segmented_text'].tolist()
        vocab, word_freq = self.build_vocabulary(texts, max_vocab_size=20000)
        
        # 6. 保存词汇表（如果指定了路径）
        if save_vocab_path:
            vocab_dir = Path(save_vocab_path).parent
            vocab_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存词汇表
            with open(save_vocab_path, 'wb') as f:
                pickle.dump(vocab, f)
            
            # 保存词频统计
            freq_path = save_vocab_path.replace('.pkl', '_freq.json')
            with open(freq_path, 'w', encoding='utf-8') as f:
                json.dump(word_freq, f, ensure_ascii=False, indent=2)
            
            print(f"  词汇表已保存: {save_vocab_path}")
            print(f"  词频统计已保存: {freq_path}")
        
        print("\n" + "="*60)
        print(f"预处理完成!")
        print(f"原始数据: {original_len:,} 条 → 处理后: {len(processed_df):,} 条")
        print("="*60)
        
        return processed_df, vocab
    
    def balance_jd_dataset(self, df, label_column='label'):
        """平衡京东评论数据集"""
        # 计算每个类别的数量
        class_counts = df[label_column].value_counts()
        
        if len(class_counts) < 2:
            print("  只有1个类别，无需平衡")
            return df
        
        print(f"  平衡前类别分布: {dict(class_counts)}")
        
        # 找到最小类别大小
        min_size = class_counts.min()
        
        balanced_dfs = []
        for label in df[label_column].unique():
            class_df = df[df[label_column] == label]
            
            if len(class_df) > min_size:
                # 下采样
                class_df_balanced = resample(
                    class_df,
                    replace=False,
                    n_samples=min_size,
                    random_state=42
                )
            else:
                # 上采样
                class_df_balanced = resample(
                    class_df,
                    replace=True,
                    n_samples=min_size,
                    random_state=42
                )
            
            balanced_dfs.append(class_df_balanced)
        
        # 合并并打乱
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 验证平衡结果
        balanced_counts = balanced_df[label_column].value_counts()
        print(f"  平衡后类别分布: {dict(balanced_counts)}")
        
        return balanced_df
    
    def save_preprocessor(self, save_path):
        """保存预处理器状态"""
        state = {
            'use_jieba': self.use_jieba,
            'jd_stop_words': list(self.jd_stop_words),
            'class_name': self.__class__.__name__
        }
        
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"预处理器状态已保存: {save_path}")
    
    @classmethod
    def load_preprocessor(cls, load_path):
        """加载预处理器状态"""
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        preprocessor = cls(
            use_jieba=state['use_jieba'],
            custom_stop_words=set(state['jd_stop_words'])
        )
        
        print(f"预处理器状态已加载: {load_path}")
        return preprocessor


# 便捷函数
def quick_preprocess(text, remove_stopwords=True):
    """
    快速预处理单条文本
    
    Args:
        text: 原始文本
        remove_stopwords: 是否移除停用词
        
    Returns:
        预处理后的文本
    """
    preprocessor = DataPreprocessor(use_jieba=True)
    cleaned = preprocessor.clean_jd_text(text)
    segmented = preprocessor.segment_jd_text(cleaned, remove_stopwords=remove_stopwords)
    return segmented


def batch_preprocess(texts, remove_stopwords=True):
    """
    批量预处理文本
    
    Args:
        texts: 文本列表
        remove_stopwords: 是否移除停用词
        
    Returns:
        预处理后的文本列表
    """
    preprocessor = DataPreprocessor(use_jieba=True)
    results = []
    
    for text in texts:
        cleaned = preprocessor.clean_jd_text(text)
        segmented = preprocessor.segment_jd_text(cleaned, remove_stopwords=remove_stopwords)
        results.append(segmented)
    
    return results


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    sample_data = pd.DataFrame({
        'sentence': [
            '质量很好，物流很快，非常满意！京东的服务就是好！',
            '质量很差，用了两天就坏了，不推荐购买。',
            '包装完好，发货速度快，服务态度好，会再次购买。',
            '客服态度恶劣，问题完全没有解决，非常失望。',
            '物美价廉，性价比很高，物流速度真是一流。'
        ],
        'label': [1, 0, 1, 0, 1]  # 1:正面, 0:负面
    })
    
    print("测试数据预处理...")
    print("="*60)
    
    # 创建预处理器
    preprocessor = DataPreprocessor(use_jieba=True)
    
    # 预处理数据
    processed_df, vocab = preprocessor.preprocess_jd_dataset(
        sample_data,
        text_column='sentence',
        label_column='label',
        balance=True,
        save_vocab_path='test_vocab.pkl'
    )
    
    print("\n处理结果:")
    print(processed_df[['sentence', 'segmented_text', 'label']].head())
    
    # 测试文本转换
    print("\n测试文本转换:")
    test_text = "京东的物流速度真是一流，晚上下单第二天就到！"
    cleaned = preprocessor.clean_jd_text(test_text)
    segmented = preprocessor.segment_jd_text(cleaned)
    indices = preprocessor.text_to_indices(segmented, vocab)
    
    print(f"原始文本: {test_text}")
    print(f"清理后: {cleaned}")
    print(f"分词后: {segmented}")
    print(f"索引: {indices[:20]}...")
    
    # 测试便捷函数
    print("\n测试便捷函数:")
    quick_result = quick_preprocess("质量很好，价格实惠！")
    print(f"快速预处理: {quick_result}")
    
    print("\n✅ 所有测试完成!")