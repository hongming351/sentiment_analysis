
"""
BERT集成模型情感分析器
支持用户输入评论进行情感分析预测
"""

import os
import sys
import torch
import json
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import re
from datetime import datetime
import argparse

class BERTSentimentAnalyzer:
    """BERT集成模型情感分析器"""
    
    def __init__(self, ensemble_model_path='models/bert_model/bert_true_ensemble_model_cv.pt', 
                 device=None):
        """
        初始化BERT情感分析器
        
        Args:
            ensemble_model_path: 集成模型文件路径
            device: 计算设备 (默认自动选择)
        """
        self.ensemble_model_path = ensemble_model_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ensemble_data = None
        self.tokenizer = None
        self.models = []
        self.max_length = 128
        
        print(f"🚀 初始化BERT情感分析器")
        print(f"📱 使用设备: {self.device}")
        print(f"🔄 加载集成模型: {ensemble_model_path}")
        
        self._load_model()
    
    def _load_model(self):
        """加载集成模型"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.ensemble_model_path):
                raise FileNotFoundError(f"集成模型文件不存在: {self.ensemble_model_path}")
            
            # 加载集成模型数据
            print("📊 加载集成模型数据...")
            self.ensemble_data = torch.load(self.ensemble_model_path, 
                                          map_location='cpu', 
                                          weights_only=False)
            
            print(f"✅ 集成模型版本: {self.ensemble_data.get('version', 'N/A')}")
            print(f"📅 创建日期: {self.ensemble_data.get('created_date', 'N/A')}")
            
            # 加载tokenizer
            print("🔤 加载BERT分词器...")
            tokenizer_info = self.ensemble_data.get('tokenizer_info', 'bert-base-chinese')
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_info)
            self.max_length = self.ensemble_data.get('max_len', 128)
            
            # 预加载所有模型
            print("🤖 预加载BERT模型...")
            self.models = []
            
            model_configs = self.ensemble_data.get('model_configs', [])
            model_states = self.ensemble_data.get('models', [])
            
            for i, (state_dict, model_config) in enumerate(zip(model_states, model_configs)):
                try:
                    print(f"  加载第{i+1}个模型...")
                    model = BertForSequenceClassification.from_pretrained(
                        model_config.get('model_path', 'bert-base-chinese'),
                        num_labels=model_config.get('num_labels', 2)
                    )
                    model.load_state_dict(state_dict)
                    model.to(self.device)
                    model.eval()
                    self.models.append(model)
                except Exception as e:
                    print(f"  ❌ 第{i+1}个模型加载失败: {e}")
                    continue
            
            if not self.models:
                raise ValueError("没有成功加载任何模型")
            
            print(f"✅ 成功加载 {len(self.models)} 个BERT模型")
            
            # 显示模型性能
            performance = self.ensemble_data.get('performance', {})
            if performance:
                print(f"\n📈 模型性能:")
                print(f"  单模型准确率: {performance.get('single_model_acc', 0):.4f}")
                print(f"  集成软投票准确率: {performance.get('ensemble_soft_acc', 0):.4f}")
                print(f"  集成硬投票准确率: {performance.get('ensemble_hard_acc', 0):.4f}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def _clean_text(self, text):
        """清理文本"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def _encode_text(self, text):
        """编码文本"""
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict_single(self, text, method='soft_voting'):
        """
        预测单个文本的情感
        
        Args:
            text: 待分析的文本
            method: 集成方法 ('soft_voting' 或 'hard_voting')
            
        Returns:
            dict: 包含预测结果的字典
        """
        if not text or not text.strip():
            return {
                'text': text,
                'sentiment': '无法分析',
                'confidence': 0.0,
                'prediction': -1,
                'probabilities': [0.5, 0.5],
                'method': method,
                'error': '输入文本为空'
            }
        
        # 清理文本
        clean_text = self._clean_text(text)
        
        if not clean_text:
            return {
                'text': text,
                'sentiment': '无法分析',
                'confidence': 0.0,
                'prediction': -1,
                'probabilities': [0.5, 0.5],
                'method': method,
                'error': '文本清理后为空'
            }
        
        # 编码文本
        inputs = self._encode_text(clean_text)
        
        try:
            with torch.no_grad():
                if method == 'soft_voting':
                    return self._soft_voting_predict(inputs, clean_text)
                elif method == 'hard_voting':
                    return self._hard_voting_predict(inputs, clean_text)
                else:
                    raise ValueError(f"不支持的集成方法: {method}")
                    
        except Exception as e:
            return {
                'text': text,
                'sentiment': '预测失败',
                'confidence': 0.0,
                'prediction': -1,
                'probabilities': [0.5, 0.5],
                'method': method,
                'error': str(e)
            }
    
    def _soft_voting_predict(self, inputs, original_text):
        """软投票预测"""
        all_probs = []
        
        for model in self.models:
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs)
        
        # 平均概率
        avg_probs = torch.mean(torch.stack(all_probs), dim=0)
        prediction = torch.argmax(avg_probs, dim=1).item()
        confidence = torch.max(avg_probs).item()
        
        sentiment = "正面" if prediction == 1 else "负面"
        probs_list = avg_probs.cpu().numpy().tolist()[0]
        
        return {
            'text': original_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'prediction': prediction,
            'probabilities': probs_list,
            'method': 'soft_voting',
            'negative_prob': probs_list[0],
            'positive_prob': probs_list[1]
        }
    
    def _hard_voting_predict(self, inputs, original_text):
        """硬投票预测"""
        all_predictions = []
        
        for model in self.models:
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            all_predictions.append(prediction)
        
        # 多数票决策
        vote_0 = all_predictions.count(0)
        vote_1 = all_predictions.count(1)
        
        if vote_0 > vote_1:
            final_prediction = 0
            confidence = vote_0 / len(all_predictions)
        else:
            final_prediction = 1
            confidence = vote_1 / len(all_predictions)
        
        sentiment = "正面" if final_prediction == 1 else "负面"
        
        return {
            'text': original_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'prediction': final_prediction,
            'probabilities': [vote_0/len(all_predictions), vote_1/len(all_predictions)],
            'method': 'hard_voting',
            'votes': {'negative': vote_0, 'positive': vote_1}
        }
    
    def predict_batch(self, texts, method='soft_voting', show_progress=True):
        """批量预测"""
        results = []
        
        print(f"🔄 开始批量预测 ({len(texts)} 个文本)")
        
        for i, text in enumerate(texts):
            if show_progress and i % 10 == 0:
                print(f"  处理进度: {i+1}/{len(texts)}")
            
            result = self.predict_single(text, method)
            results.append(result)
        
        print(f"✅ 批量预测完成")
        return results
    
    def analyze_sentiment(self, text, method='soft_voting'):
        """分析文本情感（统一接口）"""
        return self.predict_single(text, method)

def print_result(result):
    """打印预测结果"""
    if 'error' in result:
        print(f"❌ 错误: {result['error']}")
        return
    
    sentiment = result['sentiment']
    confidence = result['confidence']
    method = result['method']
    
    # 根据置信度设置表情符号
    if confidence >= 0.9:
        emoji = "🟢"
    elif confidence >= 0.7:
        emoji = "🟡"
    else:
        emoji = "🔴"
    
    print(f"\n{emoji} 情感分析结果:")
    print(f"📝 文本: {result['text']}")
    print(f"😊 情感: {sentiment}")
    print(f"🎯 置信度: {confidence:.1%}")
    print(f"⚙️ 方法: {method}")
    
    # 显示概率分布
    if 'negative_prob' in result:
        print(f"📊 概率分布:")
        print(f"   负面: {result['negative_prob']:.1%}")
        print(f"   正面: {result['positive_prob']:.1%}")
    elif 'votes' in result:
        print(f"🗳️ 投票结果:")
        print(f"   负面票数: {result['votes']['negative']}")
        print(f"   正面票数: {result['votes']['positive']}")

def interactive_mode(analyzer):
    """交互模式"""
    print("\n" + "="*60)
    print("🤖 BERT情感分析器 - 评论情感评价系统")
    print("="*60)
    print("✅ 系统已就绪！请输入您的评论，我们将为您分析情感倾向")
    print("📝 输入 'help' 查看帮助 | 输入 'quit' 退出")
    print("🎯 当前使用软投票模式（推荐）")
    print("="*60)
    
    method = 'soft_voting'
    
    while True:
        try:
            user_input = input(f"\n🎯 [{method}] 请输入评论: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 感谢使用BERT情感分析器！")
                break
            
            elif user_input.lower() == 'help':
                print("\n📚 帮助信息:")
                print("  - 输入评论文本进行分析")
                print("  - 输入 'mode' 切换集成方法")
                print("  - 输入 'quit' 退出程序")
                print("\n🔍 集成方法说明:")
                print("  - soft_voting: 软投票（推荐），综合概率，精度更高")
                print("  - hard_voting: 硬投票，多数决策，稳定可靠")
                continue
            
            elif user_input.lower() == 'mode':
                if method == 'soft_voting':
                    method = 'hard_voting'
                    print("🔄 已切换到硬投票模式")
                else:
                    method = 'soft_voting'
                    print("🔄 已切换到软投票模式")
                continue
            
            elif not user_input:
                print("输入有效的文本内容")
                continue
            
            # 进行情感分析
            result = analyzer.analyze_sentiment(user_input, method)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，感谢使用！")
            break
        except Exception as e:
            print(f"\n❌ 处理过程中发生错误: {e}")

def demo_mode(analyzer):
    """演示模式"""
    print("\n" + "="*60)
    print("🎭 BERT情感分析器 - 演示模式")
    print("="*60)
    
    # 示例评论
    demo_texts = [
        "这个商品质量真的很好，非常满意！",
        "物流太慢了，等了整整一个星期",
        "性价比很高，推荐购买",
        "包装破损，商品有瑕疵",
        "服务态度差，客服不专业",
        "价格实惠，质量也不错",
        "产品设计很棒，功能强大",
        "使用体验一般，没有想象中好"
    ]
    
    print("📋 示例评论情感分析:")
    
    for i, text in enumerate(demo_texts, 1):
        print(f"\n{i}. 文本: {text}")
        result = analyzer.analyze_sentiment(text, 'soft_voting')
        print_result(result)

def batch_mode(analyzer, file_path, output_path=None):
    """批量处理模式"""
    try:
        print(f"\n" + "="*60)
        print(f"📁 批量处理模式")
        print("="*60)
        
        # 读取文件
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'text' in df.columns:
                texts = df['text'].tolist()
            elif 'sentence' in df.columns:
                texts = df['sentence'].tolist()
            else:
                raise ValueError("CSV文件需要包含'text'或'sentence'列")
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("支持的文件格式: .csv, .txt")
        
        print(f"📖 读取到 {len(texts)} 条文本")
        
        # 批量预测
        results = analyzer.predict_batch(texts, 'soft_voting')
        
        # 保存结果
        if output_path:
            if output_path.endswith('.csv'):
                result_df = pd.DataFrame(results)
                result_df.to_csv(output_path, index=False, encoding='utf-8')
            elif output_path.endswith('.json'):
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError("输出文件格式支持: .csv, .json")
            
            print(f"💾 结果已保存到: {output_path}")
        
        # 统计结果
        sentiments = [r['sentiment'] for r in results if 'sentiment' in r]
        positive_count = sentiments.count('正面')
        negative_count = sentiments.count('负面')
        
        print(f"\n📊 批量分析统计:")
        print(f"  总数: {len(sentiments)}")
        print(f"  正面: {positive_count} ({positive_count/len(sentiments)*100:.1f}%)")
        print(f"  负面: {negative_count} ({negative_count/len(sentiments)*100:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"❌ 批量处理失败: {e}")
        return []

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BERT集成模型情感分析器')
    parser.add_argument('--model', type=str, 
                       default='models/bert_model/bert_true_ensemble_model_cv.pt',
                       help='集成模型文件路径')
    parser.add_argument('--mode', type=str, choices=['interactive', 'demo', 'batch'],
                       default='interactive', help='运行模式')
    parser.add_argument('--file', type=str, help='批量处理文件路径')
    parser.add_argument('--output', type=str, help='批量处理输出文件路径')
    parser.add_argument('--device', type=str, help='计算设备')
    
    args = parser.parse_args()
    
    try:
        # 初始化分析器
        analyzer = BERTSentimentAnalyzer(args.model, args.device)
        
        # 根据模式运行
        if args.mode == 'interactive':
            interactive_mode(analyzer)
        elif args.mode == 'demo':
            demo_mode(analyzer)
        elif args.mode == 'batch':
            if not args.file:
                print("❌ 批量模式需要指定文件路径 (--file)")
                return
            batch_mode(analyzer, args.file, args.output)
        else:
            print(f"❌ 不支持的模式: {args.mode}")
            
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 如果直接运行，默认为交互模式
    if len(sys.argv) == 1:
        try:
            analyzer = BERTSentimentAnalyzer()
            interactive_mode(analyzer)
        except Exception as e:
            print(f"❌ 程序运行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        main()
