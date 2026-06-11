#!/usr/bin/env python3
"""
QuantaAlpha模型训练与导出

支持模式：
  v1.0: 三因子均值回归（Hurst + AR1 + OU）
  v3.0: 五因子多逻辑（均值回归 + 动量 + 波动率 + 量价）

使用方法：
    source venv/bin/activate
    python3 train_model_simple.py              # v1.0默认
    python3 train_model_simple.py --mode v3.0  # v3.0五因子
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


# ============================================================
# 因子表达式库
# ============================================================
FACTOR_MODES = {
    "v1.0": [
        # 因子1: Hurst_Proxy（均值回归）
        "(0 - (Log($close) - Mean(Log($close), 20)) / (Std(Log($close), 20) + 1e-8)) * Std($close/Ref($close,1)-1, 5) / (Std($close/Ref($close,1)-1, 20) + 1e-8)",
        # 因子2: AR1_Reversion（均值回归）
        "(0 - ($close - Mean($close, 20)) / (Std($close, 20) + 1e-8)) * Sign(0 - Corr($close/Ref($close,1)-1, Ref($close/Ref($close,1)-1, 1), 20))",
        # 因子3: OU_MeanReversion（均值回归）
        "(0 - ($close - Mean($close, 20)) / (Std($close, 20) + 1e-8)) * (1 - Corr($close/Ref($close,1)-1, Ref($close/Ref($close,1)-1, 1), 10))",
    ],
    "v3.0": [
        # 因子1: Hurst_Proxy（均值回归，v1.0中ICIR最高）
        "(0 - (Log($close) - Mean(Log($close), 20)) / (Std(Log($close), 20) + 1e-8)) * Std($close/Ref($close,1)-1, 5) / (Std($close/Ref($close,1)-1, 20) + 1e-8)",
        # 因子2: 时序动量（动量/趋势类，新增）
        "Ref($close, 20) / $close - 1",
        # 因子3: 趋势强度（动量/趋势类，新增）
        "(Mean($close, 5) - Mean($close, 20)) / (Std($close, 20) + 1e-8)",
        # 因子4: 波动率加速度（波动率类，新增）
        "(Std($close/Ref($close,1)-1, 5) - Std($close/Ref($close,1)-1, 20)) / (Std($close/Ref($close,1)-1, 20) + 1e-8)",
        # 因子5: 量价背离（量价类，精简版，Rank需要(feature, window)参数）
        "Rank($close / (Mean($close, 10) + 1e-8), 60) - Rank($volume / (Mean($volume, 10) + 1e-8), 60)",
    ],
    "v4.0": [
        # Phase 1从factorlib 75因子中按ICIR选优（5因子，覆盖5个驱动逻辑）
        # 因子1: Gap_Range_Mean_Reversion_5D（均值回归类，ICIR=0.1610）
        "((($open - Ref($close, 1)) / (Mean($high - $low, 10) + 1e-8) * ($high - $low) / (Mean($high - $low, 10) + 1e-8) - Mean(($open - Ref($close, 1)) / (Mean($high - $low, 10) + 1e-8) * ($high - $low) / (Mean($high - $low, 10) + 1e-8), 5)) / (Std(($open - Ref($close, 1)) / (Mean($high - $low, 10) + 1e-8) * ($high - $low) / (Mean($high - $low, 10) + 1e-8), 5) + 1e-8))",
        # 因子2: Close_Position_Deviation_Factor_10D（动量/趋势类，ICIR=0.1610）
        "Rank(Abs(($close - $low) / ($high - $low + 1e-8) - 0.5), 10)",
        # 因子3: Overnight_Intraday_Asymmetry_20D（其他类，ICIR=0.1610）
        "Corr(($open - Ref($close, 1)) / (Ref($close, 1) + 1e-8), ($close - $low) / ($high - $low + 1e-8), 20)",
        # 因子4: Turnover_Acceleration_10D_vs_20D（量价关系类，ICIR=0.1569）
        "Rank((Mean(Delta($volume, 1), 10) - Mean(Delta($volume, 1), 20)) / (Std(Delta($volume, 1), 20) + 1e-8), 60)",
        # 因子5: Abnormal_Return_Persistence_Score（波动率类，ICIR=0.1370）
        "Count(Abs(($close - Ref($close, 1)) / (Ref($close, 1) + 1e-8) - Mean(($close - Ref($close, 1)) / (Ref($close, 1) + 1e-8), 20)) > 1.5 * Std(($close - Ref($close, 1)) / (Ref($close, 1) + 1e-8), 20), 5) / 5",
    ],
}

FACTOR_NAMES = {
    "v1.0": ["Hurst_Proxy", "AR1_Reversion", "OU_MeanReversion"],
    "v3.0": ["Hurst_Proxy", "TS_Momentum", "Trend_Strength", "Vol_Acceleration", "PV_Divergence"],
    "v4.0": ["Gap_Range_MR_5D", "Close_Pos_Dev_10D", "Overnight_Asym_20D", "Turnover_Accel_10v20", "AbnRet_Persist"],
}


def _get_default_params() -> dict:
    """Default LightGBM hyperparameters (v1.0/v3.0/v4.0)."""
    return {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.05,
        'colsample_bytree': 0.8879,
        'subsample': 0.8789,
        'lambda_l1': 205.6999,
        'lambda_l2': 580.9768,
        'max_depth': 8,
        'num_leaves': 210,
        'num_threads': 20,
        'verbose': -1
    }


def load_optuna_best_config() -> dict:
    """Load the best configuration from Optuna search results."""
    # Support both source-tree and packaged layouts
    script_dir = Path(__file__).resolve().parent
    if (script_dir.parent / "docs" / "optuna_search_results" / "best_configuration.json").exists():
        # Packaged: algorithm/ → docs/optuna_search_results/
        config_file = script_dir.parent / "docs" / "optuna_search_results" / "best_configuration.json"
    else:
        # Source tree
        config_file = project_root / "optuna_search_results" / "best_configuration.json"

    if not config_file.exists():
        print(f"❌ Optuna最优配置不存在: {config_file}")
        print("请先运行: python3 train_model_optuna.py")
        sys.exit(1)

    config = json.loads(config_file.read_text())
    print(f"✅ 加载Optuna最优配置 (Trial #{config['trial_number']})")
    print(f"  Score: {config['score']:.6f}")
    print(f"  因子数: {config['n_factors']}")
    print(f"  选中因子: {config['selected_factors']}")
    return config


def load_v5_features_from_cache(optuna_config: dict) -> tuple:
    """v5.0: Load features from Phase 1 cache instead of re-querying Qlib.

    Returns (features_df, label_df, factor_names, factor_cols, selected_exprs).
    """
    # Support both source-tree and packaged layouts
    script_dir = Path(__file__).resolve().parent
    if (script_dir.parent / "docs" / "feature_cache_labeled.h5").exists():
        # Packaged: algorithm/ → check docs/
        labeled_file = script_dir.parent / "docs" / "feature_cache_labeled.h5"
    else:
        # Source tree
        labeled_file = project_root / "feature_cache_labeled.h5"

    if not labeled_file.exists():
        print(f"❌ 特征缓存不存在: {labeled_file}")
        print("请先运行 Phase 1: python3 compute_feature_cache.py")
        sys.exit(1)

    print("\nStep 1 (v5.0): 从Phase 1缓存加载特征...")
    data_df = pd.read_hdf(labeled_file, key='data')
    print(f"  ✅ 加载缓存: {data_df.shape}")

    # Get selected columns from Optuna config
    selected_cols = optuna_config.get("selected_columns", [])
    selected_names = optuna_config.get("selected_factors", [])
    selected_exprs = optuna_config.get("selected_expressions", [])

    if not selected_cols:
        print("❌ Optuna配置中无selected_columns")
        sys.exit(1)

    # Select only the chosen factor columns + label
    available_cols = [c for c in selected_cols if c in data_df.columns]
    missing_cols = [c for c in selected_cols if c not in data_df.columns]

    if missing_cols:
        print(f"  ⚠️  以下列在缓存中缺失（可能是衍生特征）: {missing_cols[:5]}")

    if 'label' not in data_df.columns:
        print("❌ 缓存中无label列")
        sys.exit(1)

    features_df = data_df[available_cols].copy()
    label_df = data_df[['label']].copy()

    print(f"  选中因子: {len(available_cols)} 个")
    for i, name in enumerate(selected_names):
        if i < len(available_cols):
            print(f"    因子{i+1} ({name}): {available_cols[i]}")

    return features_df, label_df, selected_names, available_cols, selected_exprs


def main():
    parser = argparse.ArgumentParser(description="QuantaAlpha模型训练")
    parser.add_argument("--mode", type=str, default="v1.0",
                        choices=["v1.0", "v3.0", "v4.0", "v5.0"],
                        help="因子模式: v1.0(三因子) 或 v3.0(五因子) 或 v4.0(因子库选优5因子) 或 v5.0(Optuna搜索最优)")
    parser.add_argument("--skip-rolling", action="store_true",
                        help="跳过滚动IC计算（节省时间）")
    args = parser.parse_args()

    # ================================================================
    # v5.0: 从Phase 1缓存加载（跳过Qlib重新计算）
    # ================================================================
    if args.mode == "v5.0":
        optuna_config = load_optuna_best_config()
        features_df, label_df, factor_names, factor_cols, factor_expressions = load_v5_features_from_cache(optuna_config)

        print("=" * 70)
        print(f"QuantaAlpha模型训练与导出 (v5.0模式 — Optuna最优)")
        print("=" * 70)

        # v5.0 uses cache, skip Qlib Steps 1-4
        # Go directly to Step 5: merge and clean
        print("\nStep 2 (v5.0): 准备训练数据...")
        data_df = features_df.join(label_df, how='inner').dropna()
        print(f"  有效样本: {len(data_df)}")

        dates_index = data_df.index.get_level_values(1)
        all_dates = sorted(dates_index.unique())
        n_dates = len(all_dates)
        data_start = all_dates[0] if all_dates else pd.Timestamp("N/A")
        data_end = all_dates[-1] if all_dates else pd.Timestamp("N/A")
        print(f"  数据范围: {str(data_start)[:10]} ~ {str(data_end)[:10]} ({n_dates} 交易日)")

        train_end_date = pd.Timestamp("2023-12-31")
        valid_start_date = pd.Timestamp("2024-01-01")
        valid_end_date = pd.Timestamp("2024-12-31")
        test_start_date = pd.Timestamp("2025-01-01")

        train_mask = dates_index <= train_end_date
        valid_mask = (dates_index >= valid_start_date) & (dates_index <= valid_end_date)
        test_mask = dates_index >= test_start_date

        train_df = data_df[train_mask]
        valid_df = data_df[valid_mask]
        test_df = data_df[test_mask]

        print(f"  训练集: {len(train_df)} (2016-2023)")
        print(f"  验证集: {len(valid_df)} (2024)")
        print(f"  测试集: {len(test_df)} (2025)")

        if len(valid_df) == 0:
            print("  ⚠️  验证集为空，使用训练集最后20%作为验证")
            train_end_idx = int(n_dates * 0.80)
            train_df = data_df[dates_index < all_dates[train_end_idx]]
            valid_df = data_df[dates_index >= all_dates[train_end_idx]]
            test_df = valid_df

        X_train = train_df[factor_cols].values
        y_train = train_df['label'].values
        X_valid = valid_df[factor_cols].values
        y_valid = valid_df['label'].values
        X_test = test_df[factor_cols].values
        y_test = test_df['label'].values

        factor_expressions = factor_cols  # for display/metadata
        # Build default params
        default_params = _get_default_params()

        # v5.0: apply Optuna hyperparameters
        lgb_params = optuna_config.get("lgb_params", {})
        if lgb_params:
            for k in default_params:
                if k not in ('num_threads', 'verbose', 'objective', 'metric') and k in lgb_params:
                    default_params[k] = lgb_params[k]
            print(f"  ⚙️  使用Optuna最优超参数 (Trial #{optuna_config['trial_number']})")

        params = default_params

        # v5.0: set up variables and fall through to shared training code (Step 6)
        factor_cols = factor_cols  # already set above
        # Skip to Step 6 by setting up the same variables the standard path uses
        # (factor_expressions, factor_names, factor_cols, X_train, etc. already set)
        # Continue below to shared training code
        # v5.0 skips the standard Qlib path below via sys.exit after training

    if args.mode != "v5.0":
        # Standard path: Steps 1-5 with Qlib
        factor_expressions = FACTOR_MODES[args.mode]
        factor_names = FACTOR_NAMES[args.mode]

        print("=" * 70)
        print(f"QuantaAlpha模型训练与导出 ({args.mode}模式)")
        print("=" * 70)

        # Step 1: Init Qlib
        print("\nStep 1: 初始化Qlib...")
        import qlib
        provider_uri = Path.home() / ".qlib/qlib_data/cn_data"
        qlib.init(provider_uri=str(provider_uri), region='cn')
        print(f"✅ Qlib初始化: {provider_uri}")

        # Step 2: Print factor expressions
        print("\nStep 2: 因子表达式...")
        print(f"  模式: {args.mode} ({len(factor_expressions)}因子)")
        for i, (name, expr) in enumerate(zip(factor_names, factor_expressions), 1):
            print(f"  因子{i} ({name}): {expr[:80]}...")

        # Step 3: Compute features
        print("\nStep 3: 计算特征数据...")
        from qlib.data import D

        from qlib.data import D as data_api
        start_time = "2016-01-01"
        end_time = "2025-12-26"

        instruments_obj = data_api.instruments('csi300')
        instruments = data_api.list_instruments(
            instruments=instruments_obj,
            start_time=start_time,
            end_time=end_time,
            as_list=True,
        )

        print(f"  标的: CSI300 ({len(instruments)} stocks)")
        print(f"  时间: {start_time} ~ {end_time}")

        features_df = D.features(instruments, factor_expressions, start_time=start_time, end_time=end_time)
        features_df.columns = [f'factor_{i+1}' for i in range(len(factor_expressions))]

        print(f"✅ 特征数据: {features_df.shape}")
        print(f"  样本数: {len(features_df)}")
        print(f"  NaN比例: {features_df.isnull().sum().sum() / features_df.size * 100:.2f}%")

        # Step 4: Compute label
        print("\nStep 4: 计算标签...")
        label_expr = "Ref($close, -2)/$close - 1"
        label_df = D.features(instruments, [label_expr], start_time=start_time, end_time=end_time)
        label_df.columns = ['label']

        print(f"✅ 标签数据: {label_df.shape}")

        # Step 5: Merge and clean
        print("\nStep 5: 准备训练数据...")
        data_df = features_df.join(label_df, how='inner').dropna()

        print(f"  有效样本: {len(data_df)}")

        dates_index = data_df.index.get_level_values(1)
        all_dates = sorted(dates_index.unique())
        n_dates = len(all_dates)
        data_start = all_dates[0]
        data_end = all_dates[-1]
        print(f"  数据范围: {str(data_start)[:10]} ~ {str(data_end)[:10]} ({n_dates} 交易日)")

        train_end_date = pd.Timestamp("2023-12-31")
        valid_start_date = pd.Timestamp("2024-01-01")
        valid_end_date = pd.Timestamp("2024-12-31")
        test_start_date = pd.Timestamp("2025-01-01")

        train_mask = dates_index <= train_end_date
        valid_mask = (dates_index >= valid_start_date) & (dates_index <= valid_end_date)
        test_mask  = dates_index >= test_start_date

        train_df = data_df[train_mask]
        valid_df = data_df[valid_mask]
        test_df = data_df[test_mask]

        print(f"  训练集: {len(train_df)} (2016-01-01 ~ 2023-12-31, 全量8年)")
        print(f"  验证集: {len(valid_df)} (2024-01-01 ~ 2024-12-31)")
        print(f"  测试集: {len(test_df)} (2025-01-01 ~ {str(data_end)[:10]})")

        if len(valid_df) == 0:
            print("  ⚠️  验证集为空（数据未覆盖2024年），使用训练集最后20%作为验证")
            train_end_idx = int(n_dates * 0.80)
            train_df = data_df[dates_index < all_dates[train_end_idx]]
            valid_df = data_df[dates_index >= all_dates[train_end_idx]]
            test_df = valid_df

        factor_cols = [f'factor_{i+1}' for i in range(len(factor_expressions))]

        X_train = train_df[factor_cols].values
        y_train = train_df['label'].values

        X_valid = valid_df[factor_cols].values
        y_valid = valid_df['label'].values

        X_test = test_df[factor_cols].values
        y_test = test_df['label'].values

        params = _get_default_params()

    # ================================================================
    # Shared: Step 6 onwards (both v5.0 and standard paths converge here)
    # ================================================================
    # Step 6: 训练LightGBM模型
    print("\nStep 6: 训练LightGBM模型...")

    try:
        import lightgbm as lgb
    except ImportError:
        print("❌ lightgbm未安装")
        print("安装: pip install lightgbm")
        return 1

    import numpy as np

    # Only set default params if not already set by v5.0 path
    if 'params' not in locals():
        params = _get_default_params()

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    print("  训练参数:", {k: v for k, v in params.items() if k in ['learning_rate', 'max_depth', 'num_leaves']})

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    print(f"\n✅ 模型训练完成！最佳迭代: {model.best_iteration}")

    # Step 7: 评估模型
    print("\nStep 7: 评估模型...")

    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    train_ic = np.corrcoef(y_train_pred, y_train)[0, 1]
    valid_ic = np.corrcoef(y_valid_pred, y_valid)[0, 1]
    test_ic = np.corrcoef(y_test_pred, y_test)[0, 1]

    train_mse = np.mean((y_train_pred - y_train) ** 2)
    valid_mse = np.mean((y_valid_pred - y_valid) ** 2)
    test_mse = np.mean((y_test_pred - y_test) ** 2)

    print(f"  训练集 - IC: {train_ic:.4f}, MSE: {train_mse:.6f}")
    print(f"  验证集 - IC: {valid_ic:.4f}, MSE: {valid_mse:.6f}")
    print(f"  测试集 - IC: {test_ic:.4f}, MSE: {test_mse:.6f}")

    # Step 7b: 滚动IC验证（5窗口，诊断用，不用于训练）
    if args.skip_rolling:
        print("\nStep 7b: 跳过滚动IC计算（--skip-rolling）")
        rolling_ics = []
        rolling_windows = []
        rolling_ic_summary = []
        ic_trend = 0
        valid_rolling = []
    else:
        print("\nStep 7b: 计算滚动IC（5窗口，检测因子老化）...")

        rolling_windows = [
            ("2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
            ("2017-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
            ("2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
            ("2019-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
            ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ]

        rolling_ics = []
    for train_s, train_e, test_s, test_e in rolling_windows:
        train_s_ts, train_e_ts = pd.Timestamp(train_s), pd.Timestamp(train_e)
        test_s_ts, test_e_ts = pd.Timestamp(test_s), pd.Timestamp(test_e)

        w_train_mask = (dates_index >= train_s_ts) & (dates_index <= train_e_ts)
        w_test_mask = (dates_index >= test_s_ts) & (dates_index <= test_e_ts)

        w_train_df = data_df[w_train_mask]
        w_test_df = data_df[w_test_mask]

        if len(w_train_df) < 1000 or len(w_test_df) < 100:
            print(f"  窗口 {test_s[:4]}: 样本不足 (train={len(w_train_df)}, test={len(w_test_df)})")
            rolling_ics.append(None)
            continue

        X_w_train = w_train_df[factor_cols].values
        y_w_train = w_train_df['label'].values
        X_w_test = w_test_df[factor_cols].values
        y_w_test = w_test_df['label'].values

        # 训练临时模型
        w_train_data = lgb.Dataset(X_w_train, label=y_w_train)
        w_model = lgb.train(
            params, w_train_data, num_boost_round=500,
            valid_sets=[w_train_data], valid_names=['train'],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )

        w_pred = w_model.predict(X_w_test)
        w_ic = np.corrcoef(w_pred, y_w_test)[0, 1]
        rolling_ics.append(w_ic)
        print(f"  窗口 {test_s[:4]} (train={train_s[:4]}-{train_e[:4]}): IC={w_ic:.4f}, trees={w_model.num_trees()}")

    # 滚动IC衰减分析
    valid_rolling = [ic for ic in rolling_ics if ic is not None]
    if len(valid_rolling) >= 3:
        ic_trend = np.polyfit(range(len(valid_rolling)), valid_rolling, 1)[0]
        if ic_trend < -0.001:
            print(f"\n  ⚠️  IC衰减趋势: 斜率={ic_trend:.4f}/年，因子可能在老化")
        else:
            print(f"\n  ✅ IC趋势平稳: 斜率={ic_trend:.4f}/年，因子稳健")
        print(f"  滚动IC序列: {[f'{ic:.4f}' if ic else 'N/A' for ic in rolling_ics]}")
    rolling_ic_summary = valid_rolling

    # Step 8: 保存模型
    print("\nStep 8: 保存模型文件...")

    output_dir = project_root / "exported_models"
    output_dir.mkdir(exist_ok=True)

    # 保存为PKL格式（使用protocol=4确保兼容Python 3.4+）
    pkl_file = output_dir / "quantaalpha_model.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(model, f, protocol=4)

    print(f"✅ PKL格式: {pkl_file}")
    print(f"   大小: {pkl_file.stat().st_size / 1024:.2f} KB")

    # 保存为LightGBM原生格式
    txt_file = output_dir / "quantaalpha_model.txt"
    model.save_model(str(txt_file))

    print(f"✅ TXT格式: {txt_file}")
    print(f"   大小: {txt_file.stat().st_size / 1024:.2f} KB")

    # 保存模型元信息
    meta_file = output_dir / "model_metadata.txt"
    with open(meta_file, 'w') as f:
        f.write(f"QuantaAlpha {args.mode} 模型 ({len(factor_expressions)}因子)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"训练时间: {str(data_start)[:10]} ~ {str(data_end)[:10]}\n")
        f.write(f"训练样本: {len(train_df)}\n")
        f.write(f"验证样本: {len(valid_df)}\n")
        f.write(f"测试样本: {len(test_df)}\n\n")
        f.write(f"Split方案: 年份基准 (train=2016-2023, valid=2024, test=2025)\n\n")
        f.write(f"性能指标:\n")
        f.write(f"  训练集IC: {train_ic:.4f}\n")
        f.write(f"  验证集IC: {valid_ic:.4f}\n")
        f.write(f"  测试集IC: {test_ic:.4f}\n\n")
        f.write(f"滚动IC验证（5窗口）:\n")
        for i, (ic_val, (train_s, train_e, test_s, test_e)) in enumerate(zip(rolling_ics, rolling_windows)):
            ic_str = f"{ic_val:.4f}" if ic_val else "N/A"
            f.write(f"  窗口{i+1}: train={train_s[:4]}-{train_e[:4]} -> test={test_s[:4]}: IC={ic_str}\n")
        if valid_rolling:
            f.write(f"  IC衰减斜率: {ic_trend:.4f}/年\n")
        f.write("\n")
        f.write(f"因子定义:\n")
        for i, expr in enumerate(factor_expressions, 1):
            f.write(f"  因子{i}: {expr}\n")

    print(f"✅ 元信息: {meta_file}")

    # Step 9: 创建聚宽加载脚本
    print("\nStep 9: 创建聚宽加载脚本...")

    # 生成聚宽脚本（根据模式生成对应的因子计算函数）
    n_factors = len(factor_expressions)
    factor_init_code = ""
    if args.mode == "v3.0":
        factor_init_code = f'''    # v3.0 五因子（均值回归 + 动量 + 波动率 + 量价）
    try:
        # 因子1: Hurst_Proxy（均值回归）
        log_close = np.log(close)
        zscore1 = (log_close[-1] - np.mean(log_close[-20:])) / (np.std(log_close[-20:]) + 1e-8)
        returns = np.diff(close) / close[:-1]
        vol_short = np.std(returns[-5:])
        vol_long = np.std(returns[-20:])
        factor1 = -zscore1 * (vol_short / (vol_long + 1e-8))

        # 因子2: 时序动量
        if len(close) >= 20:
            factor2 = close[-20] / close[-1] - 1
        else:
            factor2 = 0

        # 因子3: 趋势强度
        if len(close) >= 20:
            ma5 = np.mean(close[-5:])
            ma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            factor3 = (ma5 - ma20) / (std20 + 1e-8)
        else:
            factor3 = 0

        # 因子4: 波动率加速度
        if len(returns) >= 20:
            vol_5d = np.std(returns[-5:])
            vol_20d = np.std(returns[-20:])
            factor4 = (vol_5d - vol_20d) / (vol_20d + 1e-8)
        else:
            factor4 = 0

        # 因子5: 量价背离
        if len(close) >= 10:
            close_rank = np.searchsorted(np.sort(close[-10:]), close[-1]) / 10
            vol_data = get_price(stock, end_date=date, count=10, frequency='1d', fields=['volume'])
            if vol_data is not None and len(vol_data) >= 10:
                vol_vals = vol_data['volume'].values
                vol_rank = np.searchsorted(np.sort(vol_vals[-10:]), vol_vals[-1]) / 10
                factor5 = close_rank - vol_rank
            else:
                factor5 = 0
        else:
            factor5 = 0

        return np.array([factor1, factor2, factor3, factor4, factor5])
    except:
        return None'''
    elif args.mode == "v4.0":
        factor_init_code = f'''    # v4.0 五因子（因子库选优，覆盖5个驱动逻辑）
    try:
        returns = np.diff(close) / close[:-1]

        # 因子1: Gap_Range_Mean_Reversion_5D（均值回归类，ICIR=0.1610）
        if len(close) >= 10:
            gap = (open_price[-1] - close[-2]) / (np.mean(high[-10:] - low[-10:]) + 1e-8)
            range_ratio = (high[-1] - low[-1]) / (np.mean(high[-10:] - low[-10:]) + 1e-8)
            raw = gap * range_ratio
            raw_hist = [(open_price[i] - close[i-1]) / (np.mean(high[i-10:i] - low[i-10:i]) + 1e-8) * (high[i] - low[i]) / (np.mean(high[i-10:i] - low[i-10:i]) + 1e-8) for i in range(max(1, len(close)-10), len(close))]
            if len(raw_hist) >= 5:
                factor1 = (raw - np.mean(raw_hist[-5:])) / (np.std(raw_hist[-5:]) + 1e-8)
            else:
                factor1 = 0
        else:
            factor1 = 0

        # 因子2: Close_Position_Deviation_Factor_10D（动量/趋势类，ICIR=0.1610）
        if len(close) >= 10:
            close_pos = (close[-1] - low[-1]) / (high[-1] - low[-1] + 1e-8)
            deviation = abs(close_pos - 0.5)
            dev_hist = [abs((close[i] - low[i]) / (high[i] - low[i] + 1e-8) - 0.5) for i in range(max(1, len(close)-10), len(close))]
            factor2 = sum(1 for d in dev_hist if d < deviation) / len(dev_hist) if dev_hist else 0.5
        else:
            factor2 = 0

        # 因子3: Overnight_Intraday_Asymmetry_20D（其他类，ICIR=0.1610）
        if len(close) >= 20:
            overnight_ret = [(open_price[i] - close[i-1]) / (close[i-1] + 1e-8) for i in range(1, len(close))]
            intraday_pos = [(close[i] - low[i]) / (high[i] - low[i] + 1e-8) for i in range(len(close))]
            n = min(20, len(overnight_ret), len(intraday_pos))
            if n >= 5:
                o = overnight_ret[-n:]
                i = intraday_pos[-n:]
                mean_o = np.mean(o)
                mean_i = np.mean(i)
                cov = np.mean([(o[j]-mean_o)*(i[j]-mean_i) for j in range(n)])
                std_o = np.std(o)
                std_i = np.std(i)
                factor3 = cov / (std_o * std_i + 1e-8)
            else:
                factor3 = 0
        else:
            factor3 = 0

        # 因子4: Turnover_Acceleration_10D_vs_20D（量价关系类，ICIR=0.1569）
        # Qlib: Rank((Mean(Delta($volume,1),10)-Mean(Delta($volume,1),20))/(Std(Delta($volume,1),20)+1e-8), 60)
        if len(volume) >= 80:
            vol_delta = np.diff(volume)
            accel_series = []
            for t in range(20, len(vol_delta) + 1):
                w = vol_delta[t-20:t]
                m10 = np.mean(w[-10:])
                m20 = np.mean(w)
                s20 = np.std(w)
                accel_series.append((m10 - m20) / (s20 + 1e-8))
            accel_history = accel_series[-60:] if len(accel_series) >= 60 else accel_series
            accel_cur = accel_series[-1]
            factor4 = sum(1 for a in accel_history if a < accel_cur) / len(accel_history) if accel_history else 0.5
        else:
            factor4 = 0

        # 因子5: Abnormal_Return_Persistence_Score（波动率类，ICIR=0.1370）
        if len(returns) >= 20:
            mean_ret = np.mean(returns[-20:])
            std_ret = np.std(returns[-20:])
            threshold = 1.5 * std_ret
            abnormal_count = sum(1 for r in returns[-5:] if abs(r - mean_ret) > threshold)
            factor5 = abnormal_count / 5.0
        else:
            factor5 = 0

        return np.array([factor1, factor2, factor3, factor4, factor5])
    except:
        return None'''
    elif args.mode == "v5.0":
        # v5.0: 动态因子（Optuna搜索最优）— 不自动生成聚宽脚本
        # ⚠️  v5.0因子为动态搜索，每个因子的Qlib表达式不同，需手动实现
        # 禁止导出placeholder脚本，避免部署错误
        optuna_config = load_optuna_best_config()
        selected_names = optuna_config.get("selected_factors", [])
        print(f"\n⚠️  v5.0模式：跳过聚宽脚本生成（需手动实现{len(selected_names)}个因子）")
        print("  原因: Optuna搜索的因子表达式需手动验证后实现为Python代码")
        print("  参考: exported_models/load_model_joinquant.py (v4.0示例)")
        jk_script = None  # skip
    else:
        factor_init_code = '''    try:
        # 因子1: Hurst Proxy
        log_close = np.log(close)
        zscore1 = (log_close[-1] - np.mean(log_close[-20:])) / (np.std(log_close[-20:]) + 1e-8)
        returns = np.diff(close) / close[:-1]
        vol_short = np.std(returns[-5:])
        vol_long = np.std(returns[-20:])
        factor1 = -zscore1 * (vol_short / (vol_long + 1e-8))

        # 因子2: AR1 Reversion
        zscore2 = (close[-1] - np.mean(close[-20:])) / (np.std(close[-20:]) + 1e-8)
        if len(returns) >= 20:
            corr = np.corrcoef(returns[-20:-1], returns[-19:])[0, 1]
        else:
            corr = 0
        factor2 = -zscore2 * np.sign(-corr)

        # 因子3: OU MeanReversion
        if len(returns) >= 10:
            corr_short = np.corrcoef(returns[-10:-1], returns[-9:])[0, 1]
        else:
            corr_short = 0
        factor3 = -zscore2 * (1 - corr_short)

        return np.array([factor1, factor2, factor3])
    except:
        return None'''

    # Step 9: 创建聚宽加载脚本（v5.0跳过，需手动实现）
    print("\nStep 9: 创建聚宽加载脚本...")

    if args.mode == "v5.0":
        print("  ⏭️  v5.0模式：跳过聚宽脚本生成（因子表达式需手动实现）")
    else:
        jk_script = output_dir / "load_model_joinquant.py"
        with open(jk_script, 'w') as f:
            f.write(f'''"""
聚宽平台 - 加载QuantaAlpha模型 ({args.mode}, {n_factors}因子)

前提条件：
1. 将 quantaalpha_model.txt 上传到聚宽平台（企业版）
2. 聚宽企业版支持文件上传和lightgbm库

使用步骤：
1. 上传 quantaalpha_model.txt 文件
2. 复制本脚本到策略代码
3. 运行回测

注意：聚宽平台不支持直接 open() 读取文件，
      必须使用 read_file() + tempfile + lgb.Booster 方式加载 .txt 模型。
"""

import lightgbm as lgb
import tempfile
import os
import numpy as np

BENCHMARK = "000300.XSHG"
INIT_CAPITAL = 1000000
TOP_K = 50
N_FACTORS = {n_factors}
MODEL_FILE = "quantaalpha_model.txt"

def load_model_official():
    """使用聚宽官方 API 加载 LightGBM .txt 模型"""
    import contextlib
    model_bytes = read_file(MODEL_FILE)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='wb')
    tmp_path = tmp.name
    try:
        with tmp:
            tmp.write(model_bytes)
            tmp.flush()
        model = lgb.Booster(model_file=tmp_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
    return model

def initialize(context):
    """初始化"""
    g.stock_pool = get_index_stocks(BENCHMARK)
    set_benchmark(BENCHMARK)
    set_option('use_real_price', True)

    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=0.0003, close_commission=0.0003,
        min_commission=5
    ), type='stock')

    set_slippage(FixedSlippage(0.002))

    log.info("加载模型...")
    try:
        g.model = load_model_official()
        log.info("✅ 模型加载成功")
    except Exception as e:
        log.error(f"❌ 模型加载失败: {{e}}")
        g.model = None

    run_monthly(rebalance, 1)

def calculate_factors(stock, date):
    """计算{n_factors}因子"""
    df = get_price(stock, end_date=date, count=100, frequency='1d',
                   fields=['open', 'high', 'low', 'close', 'volume'])
    if df is None or len(df) < 80:
        return None

    close = df['close'].values
    open_price = df['open'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

{factor_init_code}

def rebalance(context):
    """每月调仓"""
    if g.model is None:
        return

    log.info(f"调仓: {{context.current_dt.date()}}")

    predictions = {{}}
    for stock in g.stock_pool:
        factors = calculate_factors(stock, context.current_dt)
        if factors is not None:
            pred = g.model.predict(factors.reshape(1, -1))[0]
            predictions[stock] = pred

    if len(predictions) < TOP_K:
        return

    top_stocks = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    top_stocks = [s[0] for s in top_stocks]

    for stock in context.portfolio.positions:
        if stock not in top_stocks:
            order_target(stock, 0)

    weight = 1.0 / TOP_K
    for stock in top_stocks:
        order_target_value(stock, context.portfolio.total_value * weight)

    log.info(f"调仓完成")

def handle_data(context, data):
    pass

def after_code_changed(context):
    """代码修改后重新加载模型（聚宽热更新回调）"""
    log.info("代码已更新，重新加载模型...")
    try:
        g.model = load_model_official()
        log.info("✅ 模型重新加载成功")
    except Exception as e:
        log.error(f"❌ 模型重新加载失败: {{e}}")
        g.model = None
''')

        print(f"✅ 聚宽脚本: {jk_script}")
        print(f"  模式: {args.mode}, {n_factors}因子")

    # 完成
    print("\n" + "=" * 70)
    print("✅ 模型导出完成！")
    print("=" * 70)

    print(f"\n📦 生成的文件（exported_models/）:")
    print(f"  1. quantaalpha_model.txt     - LightGBM原生格式（⭐ 主要生产格式，上传聚宽）")
    print(f"  2. quantaalpha_model.pkl     - pickle备用格式（本地验证用）")
    print(f"  3. model_metadata.txt        - 模型元信息")
    if args.mode != "v5.0":
        print(f"  4. load_model_joinquant.py   - 聚宽加载脚本（{args.mode}, {n_factors}因子）")
    else:
        print(f"  4. load_model_joinquant.py   - ⚠️  需手动实现（v5.0因子表达式需验证）")

    print("\n📋 后续步骤:")
    print("  1. 运行 test_txt_loading.py 验证 .txt 格式可正常加载（⭐ 主要门控）")
    print("  2. 打开聚宽平台（企业版）")
    print("  3. 上传 quantaalpha_model.txt 文件")
    print("  4. 创建新策略，复制 load_model_joinquant.py 内容")
    print("  5. 运行回测")

    print("\n⚠️  注意:")
    print("  - 模型文件大小:", pkl_file.stat().st_size / 1024 / 1024, "MB")
    print("  - 聚宽基础版不支持上传文件")
    print("  - 需要聚宽企业版 + lightgbm库支持")
    print("  - 聚宽平台必须使用 .txt 格式（不支持 pickle + open()）")

    return 0

if __name__ == "__main__":
    sys.exit(main())
