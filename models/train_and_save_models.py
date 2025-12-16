"""
训练并保存朴素贝叶斯和SVM模型 - 优化版本
"""

import sys
import pandas as pd
import time
from pathlib import Path

# 添加当前目录到路径
sys.path.append('.')

def main():
    print("=" * 60)
    print("训练并保存朴素贝叶斯和SVM模型（优化版）")
    print("=" * 60)
    
    # 创建保存模型的目录
    nb_model_dir = Path("models/nb_model")
    svm_model_dir = Path("models/svm_model")
    nb_model_dir.mkdir(parents=True, exist_ok=True)
    svm_model_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\n1. 加载数据...")
    train_df = pd.read_csv('data/train.csv')
    dev_df = pd.read_csv('data/dev.csv')
    
    # 清理数据
    train_df = train_df.dropna(subset=['sentence', 'label'])
    dev_df = dev_df.dropna(subset=['sentence', 'label'])
    
    # 确保标签是整数
    train_df['label'] = train_df['label'].astype(int)
    dev_df['label'] = dev_df['label'].astype(int)
    
    # 使用更多数据进行训练（增加到 15000）
    sample_size = 15000
    train_sample = train_df.sample(min(sample_size, len(train_df)), random_state=42)
    dev_sample = dev_df.sample(min(3000, len(dev_df)), random_state=42)  # 增加测试数据
    
    print(f"  训练数据: {len(train_sample)} 条")
    print(f"  测试数据: {len(dev_sample)} 条")
    
    # 准备数据 - 传入原始文本，让模型类自己处理预处理
    X_train = train_sample['sentence'].tolist()
    X_test = dev_sample['sentence'].tolist()
    y_train = train_sample['label'].tolist()
    y_test = dev_sample['label'].tolist()
    
    # 训练朴素贝叶斯模型
    print("\n2. 训练朴素贝叶斯模型...")
    try:
        from models.nb_model.nb_model import NaiveBayesClassifier
        
        start_time = time.time()
        # 使用优化参数
        nb_classifier = NaiveBayesClassifier(alpha=0.5, use_complement=True)
        nb_result = nb_classifier.train_evaluate(X_train, X_test, y_train, y_test)
        nb_time = time.time() - start_time
        
        # 保存模型
        nb_model_path = nb_model_dir / "naive_bayes_model.pkl"
        nb_classifier.save(str(nb_model_path))
        
        print(f"  ✅ 朴素贝叶斯训练完成！")
        print(f"     准确率: {nb_result['accuracy']:.4f}")
        print(f"     F1分数: {nb_result['f1_score']:.4f}")
        print(f"     训练时间: {nb_time:.2f}秒")
        print(f"     模型保存到: {nb_model_path}")
        
    except Exception as e:
        print(f"  ❌ 朴素贝叶斯训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 训练SVM模型
    print("\n3. 训练SVM模型...")
    try:
        from models.svm_model.svm_model import SVMClassifier
        
        start_time = time.time()
        # 使用优化参数
        svm_classifier = SVMClassifier(C=1.0, use_linear_svc=True)
        svm_result = svm_classifier.train_evaluate(X_train, X_test, y_train, y_test)
        svm_time = time.time() - start_time
        
        # 保存模型
        svm_model_path = svm_model_dir / "svm_model.pkl"
        svm_classifier.save(str(svm_model_path))
        
        print(f"  ✅ SVM训练完成！")
        print(f"     准确率: {svm_result['accuracy']:.4f}")
        print(f"     F1分数: {svm_result['f1_score']:.4f}")
        print(f"     训练时间: {svm_time:.2f}秒")
        print(f"     模型保存到: {svm_model_path}")
        
    except Exception as e:
        print(f"  ❌ SVM训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 更新结果文件
    print("\n4. 更新结果文件...")
    try:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results = {}
        if 'nb_result' in locals():
            results['NaiveBayes'] = {
                'accuracy': nb_result['accuracy'],
                'f1_score': nb_result['f1_score'],
                'recall': nb_result['recall'],
                'precision': nb_result['precision'],
                'training_time': nb_time,
                'inference_time': nb_result.get('inference_time', 0.1),
                'model_params': nb_result.get('model_params', {})
            }
        
        if 'svm_result' in locals():
            results['SVM'] = {
                'accuracy': svm_result['accuracy'],
                'f1_score': svm_result['f1_score'],
                'recall': svm_result['recall'],
                'precision': svm_result['precision'],
                'training_time': svm_time,
                'inference_time': svm_result.get('inference_time', 0.5),
                'model_params': svm_result.get('model_params', {})
            }
        
        import json
        with open(results_dir / "model_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ 结果已保存到: {results_dir / 'model_results.json'}")
        
    except Exception as e:
        print(f"  ⚠️ 保存结果失败: {e}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    # 打印对比结果
    print("\n📊 模型性能对比:")
    print("-" * 50)
    if 'nb_result' in locals():
        print(f"朴素贝叶斯: 准确率={nb_result['accuracy']:.4f}, F1={nb_result['f1_score']:.4f}")
    if 'svm_result' in locals():
        print(f"SVM:        准确率={svm_result['accuracy']:.4f}, F1={svm_result['f1_score']:.4f}")
    print("-" * 50)
    
    print("\n模型文件位置：")
    print(f"  朴素贝叶斯: models/nb_model/naive_bayes_model.pkl")
    print(f"  SVM: models/svm_model/svm_model.pkl")

if __name__ == "__main__":
    main()
