import json
import os
from pathlib import Path

class Config:
    """配置文件管理器"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            "project": {
                "name": "智链新纪-京东评论情感分析",
                "description": "基于4种算法的情感分析系统，包含LSTM+BERT深度学习模型",
                "version": "1.0.0"
            },
            
            "data": {
                "train_path": "data/train.csv",
                "dev_path": "data/dev.csv",
                "use_original_split": True,
                "text_column": "review",
                "label_column": "sentiment",
                "test_size": 0.2,
                "random_state": 42,
                "max_length": 200,
                "min_word_freq": 3
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
                    "embedding_dim": 256,
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "dropout": 0.5,
                    "bidirectional": True,
                    "use_attention": True,
                    "attention_type": "self",
                    "num_heads": 4
                },
                "BERT": {
                    "use": True,
                    "model_name": "bert-base-chinese",
                    "max_length": 128,
                    "batch_size": 16,
                    "num_labels": 2
                }
            },
            
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "bert_learning_rate": 2e-5,
                "optimizer": "adam",
                "early_stopping_patience": 3,
                "save_best": True,
                "log_interval": 50
            },
            
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1", "confusion_matrix"],
                "generate_report": True,
                "plot_results": True,
                "save_predictions": True
            },
            
            "paths": {
                "data_dir": "data",
                "models_dir": "saved_models",
                "results_dir": "results",
                "plots_dir": "results/performance_plots",
                "logs_dir": "logs"
            },
            
            "optimization": {
                "param_grid": {
                    "LSTM": {
                        "learning_rate": [0.001, 0.0005, 0.0001],
                        "hidden_dim": [64, 128, 256],
                        "dropout": [0.3, 0.5, 0.7]
                    },
                    "BERT": {
                        "learning_rate": [2e-5, 3e-5, 5e-5],
                        "batch_size": [8, 16, 32]
                    }
                },
                "n_iter": 10,
                "cv": 3,
                "scoring": "f1"
            }
        }
    
    def save(self):
        """保存配置到文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        print(f"配置已保存到 {self.config_path}")
    
    def update(self, section: str, key: str, value):
        """更新配置"""
        if section in self.config:
            self.config[section][key] = value
        else:
            self.config[section] = {key: value}
    
    def get(self, section: str, key: str, default=None):
        """获取配置值"""
        return self.config.get(section, {}).get(key, default)
    
    def create_directories(self):
        """创建必要的目录结构"""
        paths = self.config.get("paths", {})
        
        for key, path in paths.items():
            if key.endswith("_dir"):
                Path(path).mkdir(parents=True, exist_ok=True)
                print(f"创建目录: {path}")

# 全局配置实例
config = Config()