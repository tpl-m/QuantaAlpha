#!/usr/bin/env python3
"""
Phase 1: 因子发现与选择 — QuantaAlpha

从factorlib 75个因子中，按ICIR排序 + 多样性检查，选出最优因子组合。

使用方法：
    source venv/bin/activate
    python3 factor_selector.py                     # 自动选优（ICIR > 0.12，覆盖3+驱动逻辑）
    python3 factor_selector.py --top 15            # 查看Top15
    python3 factor_selector.py --min-icir 0.10     # 降低ICIR门槛
    python3 factor_selector.py --list-categories   # 列出所有分类
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


# ============================================================
# 因子驱动逻辑分类规则
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
    """按关键词匹配因子驱动逻辑分类"""
    name_lower = name.lower()
    expr_lower = expression.lower()
    combined = f"{name_lower} {expr_lower}"

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return category
    return "其他"


def load_factor_library() -> list:
    """加载因子库，筛选有回测结果的因子，按ICIR排序"""
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

        category = classify_factor(
            info.get("factor_name", fid),
            info.get("qlib_expression", info.get("factor_expression", "")),
        )

        factors.append({
            "id": fid,
            "name": info.get("factor_name", fid),
            "IC": float(ic),
            "ICIR": float(icir),
            "AnnReturn": float(ann_return) if ann_return is not None else None,
            "MaxDrawdown": float(max_dd) if max_dd is not None else None,
            "expr": info.get("qlib_expression", info.get("factor_expression", "N/A")),
            "category": category,
        })

    factors.sort(key=lambda x: x["ICIR"], reverse=True)
    print(f"  有回测结果: {len(factors)} 个，无结果: {no_backtest} 个")
    return factors


def print_top_factors(factors: list, top_n: int = 20):
    """打印Top因子列表"""
    print(f"\n{'='*90}")
    print(f"因子库ICIR排名 Top-{top_n}")
    print(f"{'='*90}")
    print(f"{'排名':>4s}  {'因子名称':45s}  {'IC':>8s}  {'ICIR':>8s}  {'年化收益':>8s}  {'最大回撤':>8s}  {'驱动逻辑'}")
    print(f"{'-'*90}")

    for i, f in enumerate(factors[:top_n], 1):
        ann_str = f"{f['AnnReturn']:.2%}" if f["AnnReturn"] is not None else "N/A"
        dd_str = f"{f['MaxDrawdown']:.2%}" if f["MaxDrawdown"] is not None else "N/A"
        print(
            f"{i:4d}  {f['name']:45s}  {f['IC']:>8.4f}  {f['ICIR']:>8.4f}  {ann_str:>8s}  {dd_str:>8s}  {f['category']}"
        )


def print_category_stats(factors: list):
    """打印各驱动逻辑分类的统计"""
    categories = defaultdict(list)
    for f in factors:
        categories[f["category"]].append(f)

    print(f"\n{'='*70}")
    print("各驱动逻辑分类统计（按ICIR > 0.10筛选）")
    print(f"{'='*70}")

    for cat, items in sorted(categories.items(), key=lambda x: -max(f["ICIR"] for f in x[1]) if x[1] else 0):
        qualified = [f for f in items if f["ICIR"] > 0.10]
        print(f"\n  {cat}: 总计{len(items)}个，合格(ICIR>0.10){len(qualified)}个")
        for f in items[:5]:  # Top 5 per category
            ann_str = f"{f['AnnReturn']:.2%}" if f["AnnReturn"] is not None else "N/A"
            marker = " ✅" if f["ICIR"] > 0.10 else ""
            print(f"    {f['name']:40s}  IC={f['IC']:.4f}  ICIR={f['ICIR']:.4f}  Ann={ann_str}{marker}")


def select_factors(
    factors: list,
    min_icir: float = 0.12,
    max_factors: int = 5,
    min_categories: int = 3,
    max_corr: float = 0.7,
) -> dict:
    """
    按策略选择因子组合：
    - 从每个分类中选ICIR最高的1-2个
    - 覆盖 min_categories 个驱动逻辑
    - ⚠️  max_corr参数仅用于文档记录，实际相关性检查需要Qlib数据，
       由run_diversity_check()在Phase 1中执行
    """
    # 按分类分组
    categories = defaultdict(list)
    for f in factors:
        if f["ICIR"] > min_icir:
            categories[f["category"]].append(f)

    print(f"\n{'='*70}")
    print(f"因子选择策略（ICIR>{min_icir}，覆盖>={min_categories}分类，最多{max_factors}个）")
    print(f"{'='*70}")

    # 从每个分类选Top因子
    selected = []
    per_category_limit = max(1, max_factors // max(min_categories, len(categories)))

    for cat, items in sorted(categories.items(), key=lambda x: -max(f["ICIR"] for f in x[1]) if x[1] else 0):
        for f in items[:per_category_limit]:
            selected.append(f)
            print(f"  ✅ 选入 [{cat}]: {f['name']} (ICIR={f['ICIR']:.4f})")

    # 如果不足，补充其他分类的高ICIR因子
    if len(selected) < max_factors:
        remaining = [f for f in factors if f["ICIR"] > min_icir and f not in selected]
        for f in remaining:
            if len(selected) >= max_factors:
                break
            selected.append(f)
            print(f"  ✅ 补充 [{f['category']}]: {f['name']} (ICIR={f['ICIR']:.4f})")

    # 统计覆盖的分类数
    covered_categories = set(f["category"] for f in selected)
    print(f"\n  共选定 {len(selected)} 个因子，覆盖 {len(covered_categories)} 个驱动逻辑: {', '.join(covered_categories)}")

    return {
        "factors": selected,
        "categories": list(covered_categories),
        "n_factors": len(selected),
        "n_categories": len(covered_categories),
    }


def run_diversity_check(selection: dict) -> dict:
    """运行因子多样性检查（使用factor_diversity_checker）"""
    selected = selection["factors"]
    if len(selected) < 2:
        print("\n  ⚠️  因子数量<2，跳过多样性检查")
        return {"status": "skipped", "high_corr_pairs": []}

    exprs = [f["expr"] for f in selected]
    names = [f["name"] for f in selected]

    print(f"\n{'='*70}")
    print("因子多样性检查（相关性分析）")
    print(f"{'='*70}")

    # 导入多样性检查器
    from factor_diversity_checker import compute_factor_correlation, check_diversity, load_data

    try:
        factor_df = load_data(exprs, names, start_time="2020-01-01", end_time="2025-12-26")
        corr_matrix = compute_factor_correlation(factor_df)
        high_corr_pairs = check_diversity(corr_matrix, threshold=0.7)

        if high_corr_pairs:
            print(f"\n  ⚠️  发现 {len(high_corr_pairs)} 对高度相关因子 (|corr|>0.7):")
            for pair in high_corr_pairs:
                print(f"    {pair['factor1']} <-> {pair['factor2']}: {pair['correlation']:.4f}")
        else:
            print("\n  ✅ 无高度相关因子对，多样性良好")

        # 打印相关性矩阵
        print("\n  相关性矩阵:")
        for i, n1 in enumerate(names):
            row = f"    {n1:30s}"
            for j, n2 in enumerate(names):
                row += f"  {corr_matrix.iloc[i, j]:>6.3f}"
            print(row)

        return {"status": "passed", "high_corr_pairs": high_corr_pairs, "corr_matrix": corr_matrix.to_dict()}

    except Exception as e:
        print(f"\n  ⚠️  多样性检查失败（可能Qlib未初始化）: {e}")
        return {"status": "error", "error": str(e)}


def generate_selection_report(selection: dict, diversity: dict, output_dir: Path):
    """生成因子选择报告"""
    import datetime

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "selected_factors": [
            {
                "name": f["name"],
                "category": f["category"],
                "IC": f["IC"],
                "ICIR": f["ICIR"],
                "expression": f["expr"],
            }
            for f in selection["factors"]
        ],
        "coverage": {
            "n_factors": selection["n_factors"],
            "n_categories": selection["n_categories"],
            "categories": selection["categories"],
        },
        "diversity": {
            "status": diversity["status"],
            "high_corr_pairs": diversity.get("high_corr_pairs", []),
        },
    }

    output_dir.mkdir(exist_ok=True)
    report_file = output_dir / "factor_selection_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 因子选择报告已保存: {report_file}")
    return report


def print_final_result(selection: dict, diversity: dict) -> dict:
    """打印最终选定因子，返回可用于train_model_simple.py的配置"""
    factors = selection["factors"]

    print(f"\n{'='*70}")
    print("🎯 最终选定因子组合")
    print(f"{'='*70}")

    for i, f in enumerate(factors, 1):
        ann_str = f"  AnnReturn={f['AnnReturn']:.2%}" if f["AnnReturn"] is not None else ""
        dd_str = f"  MaxDD={f['MaxDrawdown']:.2%}" if f["MaxDrawdown"] is not None else ""
        print(f"\n  因子{i}: {f['name']}")
        print(f"    分类: {f['category']}")
        print(f"    IC={f['IC']:.4f}, ICIR={f['ICIR']:.4f}{ann_str}{dd_str}")
        print(f"    表达式: {f['expr']}")

    # 生成train_model_simple.py可用的配置
    print(f"\n{'='*70}")
    print("📋 train_model_simple.py 配置代码")
    print(f"{'='*70}")

    factor_names = [f["name"] for f in factors]
    factor_exprs = [f["expr"] for f in factors]

    print(f'\nFACTOR_NAMES = {json.dumps(factor_names, ensure_ascii=False, indent=4)}')
    print(f'\nFACTOR_EXPRESSIONS = {json.dumps(factor_exprs, ensure_ascii=False, indent=4)}')

    return {
        "FACTOR_NAMES": factor_names,
        "FACTOR_EXPRESSIONS": factor_exprs,
        "n_factors": len(factors),
    }


# ============================================================
# 主流程
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: 因子发现与选择")
    parser.add_argument("--top", type=int, default=20, help="显示Top N因子（默认20）")
    parser.add_argument("--min-icir", type=float, default=0.12, help="最低ICIR门槛（默认0.12）")
    parser.add_argument("--max-factors", type=int, default=5, help="最多选择因子数（默认5）")
    parser.add_argument("--min-categories", type=int, default=3, help="最少覆盖驱动逻辑数（默认3）")
    parser.add_argument("--list-categories", action="store_true", help="只列出分类统计")
    parser.add_argument("--skip-diversity", action="store_true", help="跳过多样性检查（默认执行）")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 1: 因子发现与选择")
    print("=" * 70)

    # 1. 加载因子库
    factors = load_factor_library()

    # 2. 打印Top因子
    print_top_factors(factors, top_n=args.top)

    # 3. 分类统计
    print_category_stats(factors)

    if args.list_categories:
        return 0

    # 4. 选择因子组合
    selection = select_factors(
        factors,
        min_icir=args.min_icir,
        max_factors=args.max_factors,
        min_categories=args.min_categories,
    )

    # 5. 多样性检查（默认执行，--skip-diversity可跳过）
    if args.skip_diversity:
        diversity = {"status": "skipped", "high_corr_pairs": []}
        print("  ⏭️  跳过多样性检查（--skip-diversity）")
    else:
        print("\n运行多样性检查（|corr|>0.7）...")
        diversity = run_diversity_check(selection)
        if diversity["status"] == "error":
            print(f"  ❌ 多样性检查失败: {diversity.get('error', '')}")
            sys.exit(1)

    # 6. 生成报告
    output_dir = project_root / "evaluation_reports"
    report = generate_selection_report(selection, diversity, output_dir)

    # 7. 打印最终结果
    config = print_final_result(selection, diversity)

    # 8. 保存配置供后续Phase使用
    config_file = output_dir / "selected_factors_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n📄 因子配置已保存: {config_file}（供Phase 2/3/4使用）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
