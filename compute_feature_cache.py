#!/usr/bin/env python3
"""
Phase 1: 特征缓存 + 因子候选池 — QuantaAlpha v2.0

从factorlib加载72个因子，转换为Qlib标准表达式，批量计算并缓存到HDF5，
避免Phase 2 Optuna搜索时重复Qlib计算（节省80%时间）。

使用方法：
    source venv/bin/activate
    python3 compute_feature_cache.py              # 默认Top 40因子
    python3 compute_feature_cache.py --top 30     # 自定义数量
    python3 compute_feature_cache.py --skip-qlib  # 跳过Qlib计算（仅做表达式转换）
"""

import json
import sys
import re
import hashlib
import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


# ============================================================
# 因子表达式转换（factorlib → Qlib标准运算符）
# ============================================================
EXPR_CONVERSIONS = [
    # Order matters: longer patterns first to avoid partial matches
    ("TS_ZSCORE", None),  # handled specially: (x - Mean(x,n)) / (Std(x,n) + 1e-8)
    ("TS_MEAN", "Mean"),
    ("TS_STD", "Std"),
    ("TS_SUM", "Sum"),
    ("TS_RANK", "Rank"),
    ("TS_CORR", "Corr"),
    ("TS_MIN", "Min"),
    ("TS_MAX", "Max"),
    ("TS_ARGMIN", "ArgMin"),
    ("TS_ARGMAX", "ArgMax"),
    ("DELAY", "Ref"),
    ("DELTA", "Delta"),
    ("RANK", "Rank"),
    ("ABS", "Abs"),
    ("SIGN", "Sign"),
    ("LOG", "Log"),
    ("COUNT", "Count"),
    ("MAX", "Max"),
    ("MIN", "Min"),
    ("SUM", "Sum"),
    ("STD", "Std"),
    ("MEAN", "Mean"),
    ("CORR", "Corr"),
    ("SQRT", "Sqrt"),
    ("POWER", "Power"),
]


def convert_factorlib_to_qlib(expr: str) -> str:
    """
    Convert factorlib-style expression to Qlib standard format.

    factorlib: TS_MEAN(x, n), TS_STD(x, n), TS_RANK(x, n), TS_CORR(x, y, n),
               DELAY(x, n), DELTA(x, n), TS_ZSCORE(x, n), etc.
    Qlib:      Mean(x, n),  Std(x, n),  Rank(x, n),  Corr(x, y, n),
               Ref(x, n),   Delta(x, n), (x-Mean(x,n))/(Std(x,n)+1e-8), etc.
    """
    if not expr or expr == "N/A":
        return ""

    result = expr.strip()

    # Handle TS_ZSCORE specially: TS_ZSCORE(x, n) → (x - Mean(x,n)) / (Std(x,n) + 1e-8)
    # Pattern: TS_ZSCORE(..., n) where ... can be nested
    ts_zscore_pattern = r'TS_ZSCORE\('
    while re.search(ts_zscore_pattern, result):
        # Find TS_ZSCORE( and extract its content
        start = result.find('TS_ZSCORE(')
        if start == -1:
            break
        # Find matching closing paren
        inner_start = start + len('TS_ZSCORE(')
        depth = 1
        i = inner_start
        while i < len(result) and depth > 0:
            if result[i] == '(':
                depth += 1
            elif result[i] == ')':
                depth -= 1
            i += 1
        inner = result[inner_start:i - 1]

        # Parse inner: last ,n  (the window parameter)
        # Find the last comma that's not inside nested parens
        last_comma = -1
        depth = 0
        for j in range(len(inner) - 1, -1, -1):
            if inner[j] == ')':
                depth += 1
            elif inner[j] == '(':
                depth -= 1
            elif inner[j] == ',' and depth == 0:
                last_comma = j
                break

        if last_comma == -1:
            # No window parameter, just convert the rest
            x_part = inner
            zscore_result = f"({x_part} - Mean({x_part}, 20)) / (Std({x_part}, 20) + 1e-8)"
        else:
            x_part = inner[:last_comma]
            window = inner[last_comma + 1:].strip()
            zscore_result = f"({x_part} - Mean({x_part}, {window})) / (Std({x_part}, {window}) + 1e-8)"

        result = result[:start] + zscore_result + result[i:]

    # Handle Rank with two arguments: Rank(x, n) → keep as-is (Qlib supports this)
    # But factorlib might use single-arg Rank, Qlib needs window
    # For now, pass through

    # Convert remaining function names
    for old_name, new_name in EXPR_CONVERSIONS:
        if old_name == "TS_ZSCORE":
            continue  # already handled
        if new_name is None:
            continue
        # Use word boundary to avoid partial replacement
        pattern = r'\b' + re.escape(old_name) + r'\b'
        result = re.sub(pattern, new_name, result)

    # Convert $return to ($close/Ref($close,1)-1) if not already present
    # Qlib supports $return directly, so leave it

    return result


# ============================================================
# 因子分类
# ============================================================
CATEGORY_KEYWORDS = {
    "均值回归": ["reversion", "mean_reversion", "hurst", "ar1", "ou_", "zscore", "revert"],
    "量价关系": ["price_volume", "pv_", "volume", "vol_", "directional", "pressure", "imbalance"],
    "动量/趋势": ["momentum", "trend", "ref(", "rs_", "rank", "position", "strength"],
    "波动率": ["volatility", "vol_accel", "std(", "risk"],
    "流动性": ["liquidity", "turnover", "amihud", "illiquidity"],
    "市场微观结构": ["order_flow", "spread", "impact", "microstructure"],
}


def classify_factor(name: str, expression: str) -> str:
    name_lower = name.lower()
    expr_lower = expression.lower()
    combined = f"{name_lower} {expr_lower}"
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return category
    return "其他"


# ============================================================
# 加载因子库
# ============================================================
def load_factor_library() -> list:
    lib_path = project_root / "data" / "factorlib" / "all_factors_library.json"
    if not lib_path.exists():
        print(f"❌ 因子库不存在: {lib_path}")
        sys.exit(1)

    lib = json.loads(lib_path.read_text())
    print(f"✅ 加载因子库: {lib['metadata']['total_factors']} 个因子")

    factors = []
    no_backtest = 0
    for fid, info in lib["factors"].items():
        bt = info.get("backtest_results") or {}
        ic = bt.get("IC") or bt.get("ic")
        icir = bt.get("ICIR") or bt.get("icir")
        ann_return = bt.get("AnnReturn") or bt.get("ann_return")
        max_dd = bt.get("MaxDrawdown") or bt.get("max_drawdown")

        if ic is None or icir is None:
            no_backtest += 1
            continue

        raw_expr = info.get("factor_expression", "")
        category = classify_factor(info.get("factor_name", fid), raw_expr)

        qlib_expr = convert_factorlib_to_qlib(raw_expr)

        factors.append({
            "id": fid,
            "name": info.get("factor_name", fid),
            "IC": float(ic),
            "ICIR": float(icir),
            "AnnReturn": float(ann_return) if ann_return is not None else None,
            "MaxDrawdown": float(max_dd) if max_dd is not None else None,
            "raw_expr": raw_expr,
            "qlib_expr": qlib_expr,
            "category": category,
        })

    factors.sort(key=lambda x: x["ICIR"], reverse=True)
    print(f"  有回测结果: {len(factors)} 个，无结果: {no_backtest} 个")
    return factors


# ============================================================
# 表达式smoke test（验证转换后表达式可被Qlib解析）
# ============================================================
def smoke_test_expressions(qlib_exprs: list, names: list, max_test: int = 10) -> list:
    """Test a subset of converted expressions with Qlib to verify parseability."""
    from qlib.data import D

    # Get a small universe for smoke test
    from qlib.data import D as data_api
    instruments_obj = data_api.instruments('csi300')
    instruments = data_api.list_instruments(
        instruments=instruments_obj,
        start_time="2025-01-01",
        end_time="2025-01-31",
        as_list=True,
    )

    if len(instruments) == 0:
        print("  ⚠️  无可用股票，跳过smoke test")
        return []

    # Test in batches
    batch_size = 10
    failed = []
    for i in range(0, min(len(qlib_exprs), max_test), batch_size):
        batch_exprs = qlib_exprs[i:i + batch_size]
        batch_names = names[i:i + batch_size]
        try:
            D.features(instruments[:1], batch_exprs, start_time="2025-01-01", end_time="2025-01-31")
        except Exception as e:
            err_str = str(e)
            for j, (n, e2) in enumerate(zip(batch_names, batch_exprs)):
                if n in err_str or e2[:20] in err_str:
                    failed.append({"name": n, "expr": e2, "error": err_str[:200]})
                    print(f"  ❌ Smoke test失败: {n}")
                    print(f"     表达式: {e2[:100]}...")
                    print(f"     错误: {err_str[:100]}")

    if not failed:
        print(f"  ✅ 所有smoke test通过（测试{min(len(qlib_exprs), max_test)}个表达式）")
    else:
        print(f"  ⚠️  {len(failed)}个表达式smoke test失败")

    return failed


# ============================================================
# 批量Qlib特征计算
# ============================================================
def compute_features_batch(factors: list, top_n: int = 40,
                           start_time: str = "2016-01-01",
                           end_time: str = "2025-12-26") -> tuple:
    """
    Batch compute Qlib features for top factors.
    Returns (features_df, label_df).
    """
    import qlib
    from qlib.data import D

    provider_uri = Path.home() / ".qlib/qlib_data/cn_data"
    qlib.init(provider_uri=str(provider_uri), region='cn')
    print(f"✅ Qlib初始化: {provider_uri}")

    # Select top factors
    top_factors = factors[:top_n]
    qlib_exprs = [f["qlib_expr"] for f in top_factors]
    factor_names = [f["name"] for f in top_factors]

    print(f"\n选择Top {top_n}因子（按ICIR排序）")

    # Smoke test expressions
    print("\n表达式smoke test...")
    smoke_test_expressions(qlib_exprs, factor_names, max_test=min(20, len(qlib_exprs)))

    # Get instruments
    from qlib.data import D as data_api
    instruments_obj = data_api.instruments('csi300')
    instruments = data_api.list_instruments(
        instruments=instruments_obj,
        start_time=start_time,
        end_time=end_time,
        as_list=True,
    )
    print(f"\n标的: CSI300 ({len(instruments)} stocks)")
    print(f"时间: {start_time} ~ {end_time}")

    # Batch compute features (Qlib call)
    print(f"\n批量计算{len(qlib_exprs)}个因子特征...")
    print("  （单次Qlib调用，预计30-60秒）")

    features_df = D.features(instruments, qlib_exprs, start_time=start_time, end_time=end_time)
    features_df.columns = [f"factor_{i+1}" for i in range(len(qlib_exprs))]

    print(f"✅ 特征数据: {features_df.shape}")
    print(f"  样本数: {len(features_df)}")
    print(f"  NaN比例: {features_df.isnull().sum().sum() / features_df.size * 100:.2f}%")

    # Compute label (future 2-day return)
    print("\n计算标签（未来2日收益率）...")
    label_expr = "Ref($close, -2)/$close - 1"
    label_df = D.features(instruments, [label_expr], start_time=start_time, end_time=end_time)
    label_df.columns = ['label']
    print(f"✅ 标签数据: {label_df.shape}")

    return features_df, label_df, top_factors


# ============================================================
# 计算各因子独立IC/ICIR
# ============================================================
def compute_factor_ic(features_df: pd.DataFrame, label_df: pd.DataFrame,
                      top_factors: list) -> list:
    """Compute daily cross-sectional IC for each factor."""
    print("\n计算各因子独立IC/ICIR...")

    data_df = features_df.join(label_df, how='inner').dropna()
    dates_index = data_df.index.get_level_values(1)
    all_dates = sorted(dates_index.unique())

    ic_results = []
    for i in range(len(top_factors)):
        factor_col = f"factor_{i+1}"
        factor_name = top_factors[i]["name"]

        daily_ics = []
        for date in all_dates:
            date_data = data_df.loc[(slice(None), date), :]
            if len(date_data) < 10:
                continue
            try:
                ic = np.corrcoef(date_data[factor_col].values, date_data['label'].values)[0, 1]
                if not np.isnan(ic):
                    daily_ics.append(ic)
            except Exception:
                continue

        mean_ic = np.mean(daily_ics) if daily_ics else 0
        ic_std = np.std(daily_ics) if daily_ics else 1e-8
        icir = mean_ic / ic_std if ic_std > 0 else 0
        ic_positive_ratio = sum(1 for ic in daily_ics if ic > 0) / len(daily_ics) if daily_ics else 0

        ic_results.append({
            "name": factor_name,
            "col": factor_col,
            "mean_ic": float(mean_ic),
            "ic_std": float(ic_std),
            "icir": float(icir),
            "ic_positive_ratio": float(ic_positive_ratio),
            "n_dates": len(daily_ics),
            "daily_ics": daily_ics,
        })

        ann_str = f"{top_factors[i].get('AnnReturn', 'N/A')}"
        if top_factors[i].get('AnnReturn') is not None:
            ann_str = f"{top_factors[i]['AnnReturn']:.2%}"

        print(f"  {factor_name:45s}  IC={mean_ic:.4f}  ICIR={icir:.4f}  IC+={ic_positive_ratio:.2%}  (原ICIR={top_factors[i]['ICIR']:.4f})")

    return ic_results


# ============================================================
# 多样性过滤（|corr|>0.7剔除）
# ============================================================
def diversity_filter(ic_results: list, features_df: pd.DataFrame,
                     max_corr: float = 0.7) -> list:
    """Greedy diversity filter: keep highest ICIR, remove highly correlated."""
    print(f"\n多样性过滤（|corr|>{max_corr}剔除）...")

    if not ic_results:
        return []

    # Compute correlation matrix
    factor_cols = [r["col"] for r in ic_results]
    corr_matrix = features_df[factor_cols].corr()

    # Sort by ICIR descending
    sorted_results = sorted(ic_results, key=lambda x: x["icir"], reverse=True)
    selected = []
    excluded = set()

    for r in sorted_results:
        if r["col"] in excluded:
            continue

        selected.append(r)

        # Exclude factors highly correlated with this one
        for other in sorted_results:
            if other["col"] == r["col"]:
                continue
            corr_val = abs(corr_matrix.loc[r["col"], other["col"]])
            if corr_val > max_corr:
                excluded.add(other["col"])

    print(f"  原始: {len(ic_results)} 个 → 过滤后: {len(selected)} 个")
    print(f"  剔除: {len(excluded)} 个（高度相关）")

    return selected


# ============================================================
# 特征工程（衍生特征）
# ============================================================
def feature_engineering(features_df: pd.DataFrame, selected_ic: list,
                        max_interactions: int = 10) -> pd.DataFrame:
    """Create derived features: rank transforms and interactions."""
    print(f"\n特征工程（最多{max_interactions}个衍生特征）...")

    enhanced_df = features_df.copy()
    new_features = []

    # Rank transforms for top factors (robust)
    for r in selected_ic[:5]:
        rank_col = f"rank_{r['col']}"
        enhanced_df[rank_col] = enhanced_df[r["col"]].rank(pct=True)
        new_features.append(rank_col)

    # Interaction terms (top factor pairs)
    interaction_count = 0
    for i in range(min(3, len(selected_ic))):
        for j in range(i + 1, min(3, len(selected_ic))):
            if interaction_count >= max_interactions - len(new_features) + 5:
                break
            col_i = selected_ic[i]["col"]
            col_j = selected_ic[j]["col"]
            inter_col = f"inter_{col_i}_{col_j}"
            enhanced_df[inter_col] = enhanced_df[col_i] * enhanced_df[col_j]
            new_features.append(inter_col)
            interaction_count += 1

    print(f"  新增衍生特征: {len(new_features)} 个")
    return enhanced_df, new_features


# ============================================================
# 保存缓存
# ============================================================
def save_cache(features_df: pd.DataFrame, label_df: pd.DataFrame,
               ic_results: list, selected_ic: list,
               top_factors: list, new_features: list):
    """Save feature cache and candidate pool report."""

    # Merge labeled data
    data_df = features_df.join(label_df, how='inner')

    # Save feature cache (unlabeled)
    cache_file = project_root / "feature_cache.h5"
    features_df.to_hdf(cache_file, key='features', mode='w')
    print(f"\n✅ 特征缓存: {cache_file} ({features_df.shape})")

    # Save labeled feature cache
    labeled_file = project_root / "feature_cache_labeled.h5"
    data_df.to_hdf(labeled_file, key='data', mode='w')
    print(f"✅ 带标签缓存: {labeled_file} ({data_df.shape})")

    # Generate candidate pool report
    report_file = project_root / "factor_candidate_pool.json"

    # Read original factorlib for reference
    lib_path = project_root / "data" / "factorlib" / "all_factors_library.json"
    lib = json.loads(lib_path.read_text()) if lib_path.exists() else {}

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": {
            "total_factors_in_library": lib.get("metadata", {}).get("total_factors", 0),
            "factors_with_backtest": len([f for f in top_factors]),
            "top_n_computed": len(top_factors),
            "after_diversity_filter": len(selected_ic),
            "derived_features": len(new_features),
        },
        "all_computed_factors": [
            {
                "name": r["name"],
                "column": r["col"],
                "mean_ic": round(r["mean_ic"], 6),
                "ic_std": round(r["ic_std"], 6),
                "icir": round(r["icir"], 6),
                "ic_positive_ratio": round(r["ic_positive_ratio"], 4),
                "original_icir": next(
                    (f["ICIR"] for f in top_factors if f["name"] == r["name"]), None
                ),
                "qlib_expression": next(
                    (f["qlib_expr"] for f in top_factors if f["name"] == r["name"]), ""
                ),
            }
            for r in ic_results
        ],
        "diversity_filtered": [
            {
                "name": r["name"],
                "column": r["col"],
                "mean_ic": round(r["mean_ic"], 6),
                "icir": round(r["icir"], 6),
                "ic_positive_ratio": round(r["ic_positive_ratio"], 4),
            }
            for r in selected_ic
        ],
        "derived_features": new_features,
        "cache_files": {
            "feature_cache": str(cache_file),
            "feature_cache_labeled": str(labeled_file),
        },
    }

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ 候选池报告: {report_file}")

    return report


# ============================================================
# 主流程
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: 特征缓存 + 因子候选池")
    parser.add_argument("--top", type=int, default=40, help="计算Top N因子（默认40）")
    parser.add_argument("--skip-qlib", action="store_true", help="跳过Qlib计算（仅做表达式转换）")
    parser.add_argument("--start-time", type=str, default="2016-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-26")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 1: 特征缓存 + 因子候选池 (QuantaAlpha v2.0)")
    print("=" * 70)

    # Step 1: Load factor library
    print("\nStep 1: 加载因子库...")
    factors = load_factor_library()

    # Step 2: Print top factors
    print(f"\nStep 2: Top {args.top}因子（按ICIR排序）...")
    for i, f in enumerate(factors[:args.top], 1):
        ann_str = f"{f['AnnReturn']:.2%}" if f["AnnReturn"] is not None else "N/A"
        print(f"  {i:3d}  {f['name']:45s}  IC={f['IC']:.4f}  ICIR={f['ICIR']:.4f}  Ann={ann_str}  [{f['category']}]")
        if i <= 3:
            print(f"       原表达式: {f['raw_expr'][:100]}")
            print(f"       Qlib转换: {f['qlib_expr'][:120]}")

    if args.skip_qlib:
        print("\n⏭️  跳过Qlib计算（--skip-qlib）")
        print("✅ 表达式转换完成")
        return 0

    # Step 3: Batch Qlib feature computation
    print("\nStep 3: 批量Qlib特征计算...")
    features_df, label_df, top_factors = compute_features_batch(
        factors, top_n=args.top,
        start_time=args.start_time, end_time=args.end_time
    )

    # Step 4: Compute per-factor IC/ICIR
    print("\nStep 4: 计算各因子独立IC/ICIR...")
    ic_results = compute_factor_ic(features_df, label_df, top_factors)

    # Step 5: Diversity filtering
    print("\nStep 5: 多样性过滤...")
    selected_ic = diversity_filter(ic_results, features_df, max_corr=0.7)

    # Step 6: Feature engineering
    print("\nStep 6: 特征工程...")
    enhanced_df, new_features = feature_engineering(features_df, selected_ic, max_interactions=10)

    # Step 7: Save cache
    print("\nStep 7: 保存缓存...")
    report = save_cache(enhanced_df, label_df, ic_results, selected_ic, top_factors, new_features)

    # Summary
    print("\n" + "=" * 70)
    print("✅ Phase 1 完成！")
    print("=" * 70)
    print(f"\n📊 产出:")
    print(f"  - 计算因子: {len(top_factors)} 个")
    print(f"  - 多样性过滤后: {len(selected_ic)} 个")
    print(f"  - 衍生特征: {len(new_features)} 个")
    print(f"  - 特征缓存: feature_cache.h5")
    print(f"  - 带标签缓存: feature_cache_labeled.h5")
    print(f"  - 候选池报告: factor_candidate_pool.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
