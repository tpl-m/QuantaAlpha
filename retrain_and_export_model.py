#!/usr/bin/env python3
"""
重新训练QuantaAlpha模型并导出为.pkl格式

用途：从已有的Qlib回测结果中提取数据，重新训练LightGBM模型，并导出为聚宽可用的.pkl文件

使用方法：
    python3 retrain_and_export_model.py
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def load_dataset_from_mlruns(run_dir: Path):
    """
    从MLflow run目录加载数据集

    Args:
        run_dir: MLflow run目录路径

    Returns:
        X_train, y_train, X_test, y_test
    """
    print(f"正在加载数据集从: {run_dir}")

    # 加载预测结果和标签
    pred_file = run_dir / "artifacts" / "pred.pkl"
    label_file = run_dir / "artifacts" / "label.pkl"

    if not pred_file.exists() or not label_file.exists():
        raise FileNotFoundError(f"预测或标签文件不存在")

    with open(pred_file, 'rb') as f:
        pred = pickle.load(f)

    with open(label_file, 'rb') as f:
        label = pickle.load(f)

    print(f"✅ 加载完成:")
    print(f"   预测: {type(pred)}, shape: {pred.shape if hasattr(pred, 'shape') else 'N/A'}")
    print(f"   标签: {type(label)}, shape: {label.shape if hasattr(label, 'shape') else 'N/A'}")

    return pred, label

def extract_features_from_qlib_data():
    """
    从Qlib数据源重新提取特征

    由于Qlib没有保存原始特征，需要重新计算
    """
    print("\n开始重新提取特征...")

    import qlib
    from qlib.data import D

    # 初始化Qlib
    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region='cn')

    # 定义三个因子的表达式（来自QuantaAlpha）
    factor_expressions = [
        # 因子1: Hurst_Proxy_MeanReversion_20D
        "-Zscore(Log($close), 20) * (Std($close/$close.shift(1)-1, 5) / (Std($close/$close.shift(1)-1, 20) + 1e-8))",

        # 因子2: AR1_Reversion_Speed_Signal_20D
        "-Zscore($close, 20) * Sign(-Corr($close/$close.shift(1)-1, Ref($close/$close.shift(1)-1, 1), 20))",

        # 因子3: OU_MeanReversion_ZScore_ReturnAdjusted_20D
        "-Zscore($close, 20) * (1 - Corr($close/$close.shift(1)-1, Ref($close/$close.shift(1)-1, 1), 10))"
    ]

    # 获取数据
    instruments = "csi300"
    start_time = "2016-01-01"
    end_time = "2025-12-26"

    print(f"提取特征: {start_time} ~ {end_time}")
    print(f"股票池: {instruments}")

    features_df_list = []

    for i, expr in enumerate(factor_expressions, 1):
        print(f"计算因子{i}...")
        try:
            df = D.features(
                instruments,
                [expr],
                start_time=start_time,
                end_time=end_time
            )
            df.columns = [f'factor_{i}']
            features_df_list.append(df)
        except Exception as e:
            print(f"❌ 因子{i}计算失败: {e}")
            return None

    # 合并所有因子
    features_df = pd.concat(features_df_list, axis=1)
    print(f"✅ 特征提取完成: {features_df.shape}")

    return features_df

def train_lgbmodel_with_qlib_config():
    """
    使用Qlib配置训练LightGBM模型

    返回训练好的model对象
    """
    print("\n开始训练LightGBM模型...")

    import qlib
    from qlib.data.dataset import DatasetH
    from qlib.contrib.data.handler import DataHandlerLP
    from qlib.contrib.model.gbdt import LGBModel

    # 初始化Qlib
    provider_uri = "~/.qlib/qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region='cn')

    # 读取配置文件（使用baseline配置）
    config_file = project_root / "git_ignore_folder" / "RD-Agent_workspace" / "5d4fe765e75e4971904efde022a17381" / "conf_baseline.yaml"

    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return None

    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✅ 加载配置: {config_file.name}")

    # 提取模型配置
    model_config = config['task']['model']
    dataset_config = config['task']['dataset']

    # 修改数据范围（使用三因子数据）
    # 需要先准备三因子数据...

    print("⚠️  此方法需要三因子数据已在Qlib中注册")
    print("建议使用下面的简化方法...")

    return None

def train_simple_lgb_model():
    """
    简化版：使用已计算的pred和label重新训练

    思路：虽然没有原始特征，但可以用预测值作为单一特征重新训练一个简单模型
    """
    print("\n⚠️  警告：这是一个简化的方法")
    print("由于无法获取原始特征，将使用预测值本身训练一个映射模型")
    print("此模型与原始模型不同，仅供参考\n")

    # 查找最新run
    mlruns_dir = project_root / "mlruns" / "535101104011168066"
    runs = [d for d in mlruns_dir.iterdir() if d.is_dir() and d.name != "meta.yaml"]
    latest_run = max(runs, key=lambda p: p.stat().st_mtime)

    # 加载预测和标签
    pred, label = load_dataset_from_mlruns(latest_run)

    print("\n❌ 由于Qlib的限制，无法从pred.pkl重建原始模型")
    print("解决方案：")
    print("1. 修改Qlib backtest配置，保存模型文件")
    print("2. 在聚宽平台重新训练（推荐）")

    return None

def main():
    print("=" * 70)
    print("QuantaAlpha模型重训练与导出工具")
    print("=" * 70)

    # 尝试方案1：从Qlib数据重新计算特征
    print("\n【方案1】从Qlib重新提取特征并训练")
    print("状态：需要安装Qlib依赖并配置数据源")

    # 尝试方案2：使用已有预测结果
    print("\n【方案2】从mlruns加载数据")
    model = train_simple_lgb_model()

    # 最终建议
    print("\n" + "=" * 70)
    print("📋 最终建议")
    print("=" * 70)
    print("\n由于Qlib默认不保存训练好的模型，您有以下选择：")
    print("\n1️⃣  **在聚宽平台重新训练**（推荐）")
    print("   - 使用已生成的 factor_ml_strategy_joinquant.py")
    print("   - 脚本会自动在初始化时训练模型")
    print("   - 优势：完全适配聚宽环境")
    print("\n2️⃣  **修改Qlib配置保存模型**")
    print("   - 在Qlib的model配置中添加 save_path")
    print("   - 重新运行backtest生成模型文件")
    print("   - 然后使用本脚本导出\n")

    print("💡 推荐使用方案1，在聚宽平台训练更可靠！\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
