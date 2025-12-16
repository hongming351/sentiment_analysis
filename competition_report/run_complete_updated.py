"""
完整系统运行脚本 - 使用真实LSTM模型数据
"""

import json
import subprocess
import pandas as pd
from pathlib import Path
import os

def create_config():
    """创建配置文件"""
    config = {
        "project": {
            "name": "智链新纪-京东评论情感分析",
            "description": "基于真实LSTM模型数据的对比分析系统"
        },
        "data": {
            "text_column": "sentence",
            "label_column": "label",
            "dataset_column": "dataset",
            "use_original_split": True,
            "max_length": 200,
            "balance_data": True,
            "sample_size": 5000
        },
        "models": {
            "NaiveBayes": {
                "use": True,
                "alpha": 1.0
            },
            "SVM": {
                "use": True,
                "C": 1.0,
                "kernel": "linear",
                "probability": True
            },
            "LSTM": {
                "use": True,
                "embedding_dim": 128,
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout": 0.5,
                "bidirectional": True,
                "use_attention": True,
                "attention_type": "self"
            },
            "BERT": {
                "use": True,
                "model_name": "bert-base-chinese",
                "max_length": 64,
                "batch_size": 8,
                "num_labels": 2
            }
        },
        "training": {
            "epochs": 15,
            "batch_size": 32,
            "learning_rate": 0.001,
            "bert_learning_rate": 2e-5,
            "optimizer": "adam",
            "early_stopping_patience": 5,
            "save_best": True,
            "log_interval": 50
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "generate_report": True,
            "plot_results": True,
            "save_predictions": True
        },
        "paths": {
            "data_dir": "data",
            "models_dir": "models",
            "results_dir": "results",
            "plots_dir": "results/performance_plots",
            "logs_dir": "logs"
        }
    }

    with open('config_complete_updated.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("✅ 配置文件已创建: config_complete_updated.json")
    return 'config_complete_updated.json'

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")

    deps = ['torch', 'transformers', 'sklearn', 'pandas', 'numpy', 'matplotlib', 'jieba', 'seaborn']

    missing = []
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} 未安装")
            missing.append(dep)

    if missing:
        print(f"\n⚠️  缺少依赖: {missing}")
        print("请运行: pip install " + " ".join(missing))
        return False

    return True

def run_lstm_training():
    """运行LSTM模型训练"""
    print("\n🚀 运行LSTM模型训练...")

    lstm_script = "models/lstm_model/train_lstm_global_vocab_visual_fixed.py"
    if os.path.exists(lstm_script):
        print(f"  运行命令: python {lstm_script}")
        try:
            result = subprocess.run(["python", lstm_script], capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                print("  ✅ LSTM训练成功完成")
                # 显示最后几行输出
                lines = result.stdout.split('\n')[-10:]
                for line in lines:
                    if line.strip():
                        print(f"    {line}")
                return True
            else:
                print("  ❌ LSTM训练失败")
                print(f"    错误: {result.stderr[-500:]}")
                return False
        except Exception as e:
            print(f"  ❌ LSTM训练异常: {e}")
            return False
    else:
        print(f"  ❌ 找不到LSTM训练脚本: {lstm_script}")
        return False

def run_model_comparison():
    """运行模型对比分析"""
    print("\n📊 运行模型对比分析...")

    comparison_script = "competition_report/updated_model_comparison.py"
    if os.path.exists(comparison_script):
        print(f"  运行命令: python {comparison_script}")
        try:
            result = subprocess.run(["python", comparison_script], capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                print("  ✅ 模型对比分析成功完成")
                return True
            else:
                print("  ❌ 模型对比分析失败")
                print(f"    错误: {result.stderr[-500:]}")
                return False
        except Exception as e:
            print(f"  ❌ 模型对比分析异常: {e}")
            return False
    else:
        print(f"  ❌ 找不到对比分析脚本: {comparison_script}")
        return False

def main():
    print("="*70)
    print("智链新纪比赛 - 京东评论情感分析完整系统（使用真实数据）")
    print("="*70)

    # 检查依赖
    if not check_dependencies():
        print("\n请先安装缺失的依赖")
        return

    # 检查数据
    print("\n📊 检查数据...")
    data_files = ['data/train.csv', 'data/dev.csv']
    for file in data_files:
        if Path(file).exists():
            df = pd.read_csv(file)
            print(f"  ✅ {file}: {len(df):,} 行")
        else:
            print(f"  ❌ {file}: 不存在")
            return

    # 检查LSTM模型数据
    print("\n🤖 检查LSTM模型数据...")
    lstm_data_files = [
        'models/lstm_model/lstm_training_log_fold_0.csv',
        'models/lstm_model/jd_lstm_fold_0_best_global.pt'
    ]
    lstm_data_exists = all(os.path.exists(f) for f in lstm_data_files)

    if lstm_data_exists:
        print("  ✅ 发现已训练的LSTM模型数据")
        use_existing = input("  是否使用已有的LSTM数据？(y/n): ").lower().strip()
        if use_existing == 'y':
            run_training = False
        else:
            run_training = True
    else:
        print("  ⚠️ 未发现LSTM模型数据，需要重新训练")
        run_training = True

    # 创建配置文件
    print("\n⚙️  创建配置文件...")
    config_file = create_config()

    print("\n🚀 开始运行完整系统")
    print("-" * 70)

    # 运行步骤
    steps = []

    if run_training:
        steps.append(("训练LSTM模型", "lstm_training"))
    else:
        print("▶️  跳过LSTM训练，使用已有数据")

    steps.append(("生成模型对比报告", "model_comparison"))

    for step_name, step_type in steps:
        print(f"\n▶️  {step_name}...")

        if step_type == "lstm_training":
            success = run_lstm_training()
        elif step_type == "model_comparison":
            success = run_model_comparison()

        if success:
            print(f"  ✅ {step_name} 成功")
        else:
            print(f"  ❌ {step_name} 失败")
            # 继续下一步

    print("\n" + "="*70)
    print("🎉 完整系统运行完成！")
    print("="*70)

    # 显示生成的文件
    print("\n📁 生成的主要文件:")
    output_files = [
        "models/lstm_model/lstm_training_log_fold_0.csv",
        "models/lstm_model/lstm_best_results_fold_0.csv",
        "models/lstm_model/jd_lstm_fold_0_best_global.pt",
        "models/lstm_model/visualizations/",
        "lstm_training_curves.png",
        "model_accuracy_comparison.png",
        "detailed_model_comparison.csv",
        "final_analysis_report.md"
    ]

    for file_path in output_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                print(f"  ✅ {file_path} (目录)")
            else:
                print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️ {file_path} (未生成)")

    print("\n📄 报告文件:")
    print("  - final_analysis_report.md (完整分析报告)")
    print("  - detailed_model_comparison.csv (详细对比数据)")
    print("="*70)

if __name__ == "__main__":
    main()
