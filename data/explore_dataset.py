
"""
数据集探索脚本
用于查看和了解数据集结构
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def explore_dataset():
    """探索数据集"""
    data_dir = Path("data")
    
    print("="*60)
    print("数据集探索工具")
    print("="*60)
    
    # 检查文件
    files = list(data_dir.glob("*.*"))
    print(f"数据目录下找到 {len(files)} 个文件:")
    for file in files:
        print(f"  - {file.name} ({file.stat().st_size/1024:.1f} KB)")
    
    # 加载CSV文件
    csv_files = list(data_dir.glob("*.csv"))
    for csv_file in csv_files:
        print(f"\n分析文件: {csv_file.name}")
        print("-"*40)
        
        try:
            # 尝试不同编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    print(f"  编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print("  无法读取文件，尝试跳过错误...")
                df = pd.read_csv(csv_file, encoding='utf-8', errors='ignore')
            
            # 显示基本信息
            print(f"  形状: {df.shape[0]} 行 × {df.shape[1]} 列")
            print(f"  列名: {list(df.columns)}")
            
            # 显示前几行
            print("\n  前5行数据:")
            print(df.head().to_string())
            
            # 数据类型
            print("\n  数据类型:")
            print(df.dtypes.to_string())
            
            # 缺失值
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print("\n  缺失值统计:")
                print(missing[missing > 0].to_string())
            
            # 如果有文本列，显示统计信息
            text_columns = df.select_dtypes(include=['object']).columns
            for col in text_columns:
                if len(df[col]) > 0:
                    sample_text = str(df[col].iloc[0])
                    print(f"\n  列 '{col}' 示例: {sample_text[:100]}...")
                    print(f"  唯一值数量: {df[col].nunique()}")
                    
                    # 文本长度统计
                    lengths = df[col].astype(str).str.len()
                    print(f"  文本长度 - 平均: {lengths.mean():.1f}, "
                          f"最小: {lengths.min()}, 最大: {lengths.max()}")
            
            # 如果有数值列，显示分布
            num_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for col in num_columns:
                print(f"\n  列 '{col}' 统计:")
                print(f"    唯一值: {df[col].nunique()}")
                print(f"    最小值: {df[col].min()}, 最大值: {df[col].max()}")
                print(f"    均值: {df[col].mean():.2f}, 中位数: {df[col].median():.2f}")
                
                if df[col].nunique() < 20:  # 分类变量
                    value_counts = df[col].value_counts().sort_index()
                    print(f"    值分布:")
                    for val, count in value_counts.items():
                        percentage = count / len(df) * 100
                        print(f"      {val}: {count} ({percentage:.1f}%)")
            
        except Exception as e:
            print(f"  读取文件时出错: {e}")
    
    # 加载JSON文件
    json_files = list(data_dir.glob("*.json"))
    for json_file in json_files:
        print(f"\n分析文件: {json_file.name}")
        print("-"*40)
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  JSON数据类型: {type(data)}")
            
            if isinstance(data, dict):
                print(f"  字典键: {list(data.keys())}")
                # 显示部分内容
                for key, value in list(data.items())[:5]:
                    print(f"    {key}: {str(value)[:100]}...")
            elif isinstance(data, list):
                print(f"  列表长度: {len(data)}")
                if len(data) > 0:
                    print(f"  第一个元素类型: {type(data[0])}")
                    if isinstance(data[0], dict):
                        print(f"  第一个元素键: {list(data[0].keys())}")
            
        except Exception as e:
            print(f"  读取JSON文件时出错: {e}")
    
    print("\n" + "="*60)
    print("数据集探索完成")
    print("="*60)

def create_sample_data():
    """创建示例数据集（如果数据不存在）"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    train_path = data_dir / "train.csv"
    dev_path = data_dir / "dev.csv"
    
    # 如果文件不存在，创建示例数据
    if not train_path.exists():
        print("创建示例训练数据集...")
        sample_data = {
            'review': [
                '这个商品质量很好，物流也很快，非常满意！',
                '产品质量一般，没有想象中的好，但价格便宜',
                '非常差的产品，用了两天就坏了，要求退货',
                '客服态度很好，解决问题很及时',
                '包装完好，发货速度快，商品与描述一致',
                '质量很差，不推荐购买',
                '性价比很高，物超所值',
                '物流太慢了，等了一个星期才收到',
                '商品有瑕疵，联系客服后很快解决了',
                '非常棒的一次购物体验'
            ],
            'sentiment': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(train_path, index=False, encoding='utf-8-sig')
        print(f"示例训练数据已创建: {train_path}")
    
    if not dev_path.exists():
        print("创建示例验证数据集...")
        sample_data = {
            'review': [
                '产品不错，就是价格有点贵',
                '质量太差了，完全不值这个价',
                '物流速度快，包装仔细',
                '客服回复慢，问题没有解决',
                '商品与图片不符，很失望'
            ],
            'sentiment': [1, 0, 1, 0, 0]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(dev_path, index=False, encoding='utf-8-sig')
        print(f"示例验证数据已创建: {dev_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        create_sample_data()
    else:
        explore_dataset()