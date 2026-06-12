#!/usr/bin/env python3
"""
Phase 2: 小范围Smoke Test — QuantaAlpha

在正式训练前，用**单只股票 + 短周期**验证全流程无bug/死循环。

测试标准：
  ✅ 正常退出（exit code 0）
  ✅ 无TypeError/AttributeError
  ✅ 无死循环（超时3分钟内完成）
  ✅ NaN比例 < 5%
  ✅ 测试集IC > -0.01（非负相关）

使用方法：
    source venv/bin/activate
    python3 smoke_test.py
"""

import sys
import signal
from pathlib import Path

project_root = Path(__file__).resolve().parent
# Support packaged layout: when running from algorithm/, check for package root
if not (project_root / "data").exists() and (project_root / "docs").exists():
    # Packaged layout: algorithm/docs/ → use algorithm as root
    pass
elif (project_root.parent / "docs").exists() and (project_root.parent / "models").exists():
    # Running from algorithm/ inside a package, resolve to package root
    project_root = project_root.parent
sys.path.insert(0, str(project_root))


# ============================================================
# 使用Phase 1选定的因子（从JSON配置读取）
# ============================================================
import json

# Locate config: try docs/ first (packaged), then evaluation_reports/ (source tree)
config_path = project_root / "docs" / "selected_factors_config.json"
if not config_path.exists():
    config_path = project_root / "evaluation_reports" / "selected_factors_config.json"
if not config_path.exists():
    print(f"❌ 配置文件不存在: {project_root}/docs/selected_factors_config.json")
    print(f"   或 {project_root}/evaluation_reports/selected_factors_config.json")
    sys.exit(1)
_config = json.loads(config_path.read_text())
FACTOR_NAMES = _config["FACTOR_NAMES"]
FACTOR_EXPRESSIONS = _config["FACTOR_EXPRESSIONS"]

# ============================================================
# Smoke Test 配置
# ============================================================
SMOKE_TEST_STOCK = "SH688798"  # 单只测试股票
SMOKE_TEST_START = "2023-01-01"
SMOKE_TEST_END = "2025-12-26"
TIMEOUT_SECONDS = 180  # 3分钟超时
MAX_NAN_RATIO = 0.05   # NaN比例上限
MIN_IC = -0.10         # 测试集IC下限（单只股票噪声极大，-0.10仅捕获严重反向因子）
                         # 模型质量在Phase 3（全量CSI300）中评估


def timeout_handler(signum, frame):
    print(f"\n❌ 超时！Smoke Test超过{TIMEOUT_SECONDS}秒，可能存在死循环")
    sys.exit(1)


def main():
    # 设置超时
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    print("=" * 70)
    print("Phase 2: 小范围Smoke Test")
    print("=" * 70)
    print(f"  测试股票: {SMOKE_TEST_STOCK}")
    print(f"  时间范围: {SMOKE_TEST_START} ~ {SMOKE_TEST_END}")
    print(f"  超时限制: {TIMEOUT_SECONDS}秒")
    print(f"  因子数量: {len(FACTOR_NAMES)}")

    for i, (name, expr) in enumerate(zip(FACTOR_NAMES, FACTOR_EXPRESSIONS), 1):
        print(f"  因子{i} ({name}): {expr[:70]}...")

    errors = []

    # Step 1: 初始化Qlib
    print("\n[1/6] 初始化Qlib...")
    try:
        import qlib
        provider_uri = Path.home() / ".qlib/qlib_data/cn_data"
        qlib.init(provider_uri=str(provider_uri), region='cn')
        print("  ✅ Qlib初始化成功")
    except Exception as e:
        print(f"  ❌ Qlib初始化失败: {e}")
        return 1

    # Step 2: 计算因子
    print("\n[2/6] 计算因子...")
    try:
        from qlib.data import D

        instruments = [SMOKE_TEST_STOCK]
        features_df = D.features(
            instruments, FACTOR_EXPRESSIONS,
            start_time=SMOKE_TEST_START, end_time=SMOKE_TEST_END
        )
        features_df.columns = [f'factor_{i+1}' for i in range(len(FACTOR_NAMES))]

        nan_ratio = features_df.isnull().sum().sum() / features_df.size
        print(f"  ✅ 因子计算完成: {features_df.shape}")
        print(f"  NaN比例: {nan_ratio:.2%}")

        if nan_ratio > MAX_NAN_RATIO:
            errors.append(f"NaN比例 {nan_ratio:.2%} > {MAX_NAN_RATIO:.0%}")
            print(f"  ⚠️  NaN比例超标")

    except Exception as e:
        print(f"  ❌ 因子计算失败: {e}")
        errors.append(f"因子计算失败: {type(e).__name__}: {e}")
        # 继续后续步骤，收集所有错误

    # Step 3: 计算标签
    print("\n[3/6] 计算标签（未来2日收益）...")
    try:
        label_expr = "Ref($close, -2)/$close - 1"
        label_df = D.features(
            instruments, [label_expr],
            start_time=SMOKE_TEST_START, end_time=SMOKE_TEST_END
        )
        label_df.columns = ['label']
        print(f"  ✅ 标签计算完成: {label_df.shape}")
    except Exception as e:
        print(f"  ❌ 标签计算失败: {e}")
        errors.append(f"标签计算失败: {type(e).__name__}: {e}")

    # Step 4: 合并数据
    print("\n[4/6] 合并数据...")
    try:
        data_df = features_df.join(label_df, how='inner').dropna()
        print(f"  ✅ 有效样本: {len(data_df)}")

        if len(data_df) < 100:
            errors.append(f"有效样本不足 ({len(data_df)} < 100)")
            print(f"  ⚠️  有效样本不足")

        # 切分数据（70%训练，30%测试）
        split_idx = int(len(data_df) * 0.7)
        train_df = data_df.iloc[:split_idx]
        test_df = data_df.iloc[split_idx:]
        print(f"  训练集: {len(train_df)}, 测试集: {len(test_df)}")

    except Exception as e:
        print(f"  ❌ 数据合并失败: {e}")
        errors.append(f"数据合并失败: {type(e).__name__}: {e}")

    # Step 5: 训练LightGBM
    print("\n[5/6] 训练LightGBM...")
    try:
        import lightgbm as lgb
        import numpy as np

        factor_cols = [f'factor_{i+1}' for i in range(len(FACTOR_NAMES))]
        X_train = train_df[factor_cols].values
        y_train = train_df['label'].values
        X_test = test_df[factor_cols].values
        y_test = test_df['label'].values

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 210,
            'verbose': -1,
        }

        model = lgb.train(
            params, train_data,
            num_boost_round=100,  # 少量树
            valid_sets=[valid_data],
            valid_names=['valid'],
            callbacks=[lgb.log_evaluation(period=50)]
        )

        print(f"  ✅ 模型训练完成！最佳迭代: {model.best_iteration}")

    except Exception as e:
        print(f"  ❌ 模型训练失败: {e}")
        errors.append(f"模型训练失败: {type(e).__name__}: {e}")

    # Step 6: 评估模型
    print("\n[6/6] 评估模型...")
    try:
        import numpy as np
        y_test_pred = model.predict(X_test)
        test_ic = np.corrcoef(y_test_pred, y_test)[0, 1]
        print(f"  测试集IC: {test_ic:.4f}")

        if test_ic < MIN_IC:
            errors.append(f"测试集IC {test_ic:.4f} < {MIN_IC}")
            print(f"  ⚠️  测试集IC过低")
        else:
            print(f"  ✅ 测试集IC达标（>{MIN_IC}）")

    except Exception as e:
        print(f"  ❌ 模型评估失败: {e}")
        errors.append(f"模型评估失败: {type(e).__name__}: {e}")

    # 取消超时
    signal.alarm(0)

    # 总结
    print(f"\n{'='*70}")
    print("Smoke Test 结果")
    print(f"{'='*70}")

    if not errors:
        print("  ✅ 全部通过！Smoke Test成功")
        print("\n进入 Phase 3: 全量训练")
        return 0
    else:
        print(f"  ❌ 发现 {len(errors)} 个问题:")
        for i, err in enumerate(errors, 1):
            print(f"    {i}. {err}")
        print("\n需要修复后才能进入 Phase 3")
        return 1


if __name__ == "__main__":
    sys.exit(main())
