"""
从BERT集成模型中提取统计数据并保存为CSV格式
"""

import os
import pandas as pd
import numpy as np
import torch
import json

def extract_bert_ensemble_stats():
    """从BERT集成模型中提取统计数据并保存为CSV格式"""

    print("="*70)
    print("🔍 BERT集成模型统计数据提取工具")
    print("="*70)

    ensemble_file = 'models/bert_model/bert_true_ensemble_model_cv.pt'
    
    if not os.path.exists(ensemble_file):
        print(f"❌ 未找到集成模型文件: {ensemble_file}")
        return

    print(f"✅ 发现BERT集成模型: {ensemble_file}")

    try:
        # 加载集成模型数据
        print("\n📊 加载集成模型数据...")
        ensemble_data = torch.load(ensemble_file, map_location='cpu', weights_only=False)
        
        print(f"集成模型版本: {ensemble_data.get('version', 'N/A')}")
        print(f"创建日期: {ensemble_data.get('created_date', 'N/A')}")
        print(f"设备: {ensemble_data.get('device', 'N/A')}")
        print(f"模型类别: {ensemble_data.get('model_class', 'N/A')}")
        print(f"分词器类别: {ensemble_data.get('tokenizer_class', 'N/A')}")

        # 提取性能数据
        if 'performance' in ensemble_data:
            perf = ensemble_data['performance']
            print(f"\n📈 BERT集成模型性能:")
            print(f"   单模型准确率: {perf.get('single_model_acc', 0):.4f}")
            print(f"   集成软投票准确率: {perf.get('ensemble_soft_acc', 0):.4f}")
            print(f"   集成硬投票准确率: {perf.get('ensemble_hard_acc', 0):.4f}")
            print(f"   单模型F1分数: {perf.get('single_model_f1', 0):.4f}")
            print(f"   集成软投票F1分数: {perf.get('ensemble_soft_f1', 0):.4f}")
            print(f"   集成硬投票F1分数: {perf.get('ensemble_hard_f1', 0):.4f}")
            
            # 计算性能提升
            soft_improvement = perf.get('ensemble_soft_acc', 0) - perf.get('single_model_acc', 0)
            hard_improvement = perf.get('ensemble_hard_acc', 0) - perf.get('single_model_acc', 0)
            
            print(f"\n📊 性能提升:")
            print(f"   软投票准确率提升: +{soft_improvement*100:.2f}%")
            print(f"   硬投票准确率提升: +{hard_improvement*100:.2f}%")

            # 保存性能汇总为CSV
            performance_summary = {
                'model_type': ['BERT_Ensemble'],
                'single_model_accuracy': [perf.get('single_model_acc', 0)],
                'ensemble_soft_accuracy': [perf.get('ensemble_soft_acc', 0)],
                'ensemble_hard_accuracy': [perf.get('ensemble_hard_acc', 0)],
                'single_model_f1': [perf.get('single_model_f1', 0)],
                'ensemble_soft_f1': [perf.get('ensemble_soft_f1', 0)],
                'ensemble_hard_f1': [perf.get('ensemble_hard_f1', 0)],
                'soft_improvement': [soft_improvement],
                'hard_improvement': [hard_improvement],
                'max_length': [ensemble_data.get('max_len', 'N/A')],
                'version': [ensemble_data.get('version', 'N/A')],
                'created_date': [ensemble_data.get('created_date', 'N/A')]
            }

            performance_df = pd.DataFrame(performance_summary)
            performance_path = 'models/bert_model/models/bert_ensemble_performance.csv'
            os.makedirs(os.path.dirname(performance_path), exist_ok=True)
            performance_df.to_csv(performance_path, index=False, encoding='utf-8')
            print(f"✅ 集成模型性能汇总已保存: {performance_path}")

        # 提取各折结果
        if 'fold_results' in ensemble_data:
            fold_results = ensemble_data['fold_results']
            print(f"\n📋 各折训练结果 (共{len(fold_results)}折):")
            
            fold_data = []
            for i, result in enumerate(fold_results):
                print(f"   折{i+1}:")
                print(f"     最佳验证准确率: {result.get('best_val_acc', 0):.4f}")
                print(f"     最佳验证F1: {result.get('best_val_f1', 0):.4f}")
                print(f"     最佳epoch: {result.get('best_epoch', 0) + 1}")
                
                fold_info = {
                    'fold': i + 1,
                    'best_val_accuracy': result.get('best_val_acc', 0),
                    'best_val_f1': result.get('best_val_f1', 0),
                    'best_epoch': result.get('best_epoch', 0) + 1,
                    'best_val_loss': result.get('best_val_loss', 0),
                    'model_path': result.get('model_path', ''),
                    'transformers_path': result.get('transformers_path', '')
                }
                fold_data.append(fold_info)

            # 保存各折结果
            fold_df = pd.DataFrame(fold_data)
            fold_path = 'models/bert_model/models/bert_fold_results.csv'
            os.makedirs(os.path.dirname(fold_path), exist_ok=True)
            fold_df.to_csv(fold_path, index=False, encoding='utf-8')
            print(f"✅ 各折结果已保存: {fold_path}")

        # 提取模型配置
        if 'model_configs' in ensemble_data:
            model_configs = ensemble_data['model_configs']
            print(f"\n🔧 模型配置信息:")
            
            config_data = []
            for i, config in enumerate(model_configs):
                print(f"   折{i+1}:")
                print(f"     模型路径: {config.get('model_path', 'N/A')}")
                print(f"     标签数: {config.get('num_labels', 'N/A')}")
                print(f"     最大长度: {config.get('max_length', 'N/A')}")
                
                config_info = {
                    'fold': i + 1,
                    'model_path': config.get('model_path', ''),
                    'num_labels': config.get('num_labels', 0),
                    'max_length': config.get('max_length', 0)
                }
                config_data.append(config_info)

            # 保存模型配置
            config_df = pd.DataFrame(config_data)
            config_path = 'models/bert_model/models/bert_ensemble_configs.csv'
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            config_df.to_csv(config_path, index=False, encoding='utf-8')
            print(f"✅ 集成模型配置已保存: {config_path}")

        # 创建综合报告
        print(f"\n📊 生成综合分析报告...")
        
        if 'performance' in ensemble_data and 'fold_results' in ensemble_data:
            perf = ensemble_data['performance']
            fold_results = ensemble_data['fold_results']
            
            # 计算统计指标
            fold_accuracies = [r.get('best_val_acc', 0) for r in fold_results]
            avg_fold_acc = np.mean(fold_accuracies)
            std_fold_acc = np.std(fold_accuracies)
            max_fold_acc = np.max(fold_accuracies)
            min_fold_acc = np.min(fold_accuracies)
            
            comprehensive_report = {
                'model_family': ['BERT'],
                'ensemble_method': ['5-fold_CV'],
                'individual_folds': [len(fold_results)],
                'avg_fold_accuracy': [avg_fold_acc],
                'std_fold_accuracy': [std_fold_acc],
                'max_fold_accuracy': [max_fold_acc],
                'min_fold_accuracy': [min_fold_acc],
                'single_model_test_acc': [perf.get('single_model_acc', 0)],
                'ensemble_soft_test_acc': [perf.get('ensemble_soft_acc', 0)],
                'ensemble_hard_test_acc': [perf.get('ensemble_hard_acc', 0)],
                'soft_improvement_over_single': [perf.get('ensemble_soft_acc', 0) - perf.get('single_model_acc', 0)],
                'hard_improvement_over_single': [perf.get('ensemble_hard_acc', 0) - perf.get('single_model_acc', 0)],
                'best_individual_fold': [np.argmax(fold_accuracies) + 1],
                'training_strategy': ['Cross_Validation'],
                'ensemble_strategy': ['Soft_Voting_and_Hard_Voting'],
                'model_size': ['BERT_Base_Chinese'],
                'version': [ensemble_data.get('version', 'N/A')]
            }

            comprehensive_df = pd.DataFrame(comprehensive_report)
            comprehensive_path = 'models/bert_model/models/bert_comprehensive_analysis.csv'
            os.makedirs(os.path.dirname(comprehensive_path), exist_ok=True)
            comprehensive_df.to_csv(comprehensive_path, index=False, encoding='utf-8')
            print(f"✅ 综合分析报告已保存: {comprehensive_path}")

            print(f"\n🎯 BERT集成模型完整信息:")
            print(f"   模型类型: BERT集成模型 (5折交叉验证)")
            print(f"   集成策略: 软投票 + 硬投票")
            print(f"   最佳单折准确率: {max(fold_accuracies):.4f}")
            print(f"   平均折准确率: {avg_fold_acc:.4f} ± {std_fold_acc:.4f}")
            print(f"   集成软投票测试准确率: {perf.get('ensemble_soft_acc', 0):.4f}")
            print(f"   集成硬投票测试准确率: {perf.get('ensemble_hard_acc', 0):.4f}")

    except Exception as e:
        print(f"❌ 处理集成模型时出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ BERT集成模型数据提取完成！")
    print("="*70)

if __name__ == "__main__":
    extract_bert_ensemble_stats()
