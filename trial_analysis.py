#!/usr/bin/env python3
"""
Phase 2 补充: Optuna Trial分析 — QuantaAlpha v2.0

分析Optuna搜索结果，可视化参数重要性、收敛曲线、因子选择频率。

使用方法：
    source venv/bin/activate
    python3 trial_analysis.py                    # 默认分析
    python3 trial_analysis.py --plot-importance   # 参数重要性图
    python3 trial_analysis.py --plot-convergence  # 收敛曲线
    python3 trial_analysis.py --factor-frequency  # 因子选择频率
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def load_trial_results() -> list:
    """Load all trials from JSON."""
    results_file = project_root / "optuna_search_results" / "all_trials.json"
    if not results_file.exists():
        print(f"❌ Trial结果不存在: {results_file}")
        print("请先运行: python3 train_model_optuna.py")
        sys.exit(1)

    return json.loads(results_file.read_text())


def load_best_config() -> dict:
    """Load best configuration."""
    config_file = project_root / "optuna_search_results" / "best_configuration.json"
    if not config_file.exists():
        print(f"❌ 最优配置不存在: {config_file}")
        sys.exit(1)

    return json.loads(config_file.read_text())


def analyze_factor_frequency(trials: list, n_factors: int) -> dict:
    """Analyze how often each factor was selected across trials."""
    factor_counts = Counter()
    total_completed = 0

    for t in trials:
        attrs = t.get("user_attrs", {})
        if attrs.get("status") != "completed":
            continue
        total_completed += 1

        params = t.get("params", {})
        for i in range(n_factors):
            prob_key = f"factor_prob_{i}"
            if prob_key in params and params[prob_key] >= 0.5:
                factor_counts[i] += 1

    # Sort by frequency
    factor_freq = {}
    for idx, count in factor_counts.most_common():
        factor_freq[idx] = {
            "count": count,
            "frequency": count / total_completed if total_completed > 0 else 0,
        }

    return factor_freq, total_completed


def analyze_param_importance(trials: list) -> dict:
    """Analyze parameter importance using simple correlation method."""
    param_values = {}
    scores = []

    for t in trials:
        if t.get("state") != "COMPLETE":
            continue
        scores.append(t["value"])
        for param, value in t["params"].items():
            if param not in param_values:
                param_values[param] = []
            param_values[param].append(value)

    # Compute correlation with score for each parameter
    importance = {}
    scores_arr = np.array(scores)

    for param, values in param_values.items():
        values_arr = np.array(values, dtype=float)
        if len(values_arr) < 10:
            continue
        # Simple: use absolute correlation as importance proxy
        corr = abs(np.corrcoef(values_arr, scores_arr)[0, 1])
        importance[param] = float(corr) if not np.isnan(corr) else 0

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def analyze_convergence(trials: list) -> dict:
    """Analyze score convergence over trials."""
    best_so_far = []
    current_best = -float("inf")

    for t in sorted(trials, key=lambda x: x["number"]):
        if t.get("state") != "COMPLETE":
            continue
        if t["value"] > current_best:
            current_best = t["value"]
        best_so_far.append({
            "trial": t["number"],
            "best_so_far": current_best,
            "current": t["value"],
        })

    return {
        "n_trials": len(best_so_far),
        "final_best": current_best,
        "best_trial": best_so_far[-1] if best_so_far else None,
        "trajectory": best_so_far[-20:] if len(best_so_far) > 20 else best_so_far,  # last 20
    }


def print_summary(trials: list):
    """Print comprehensive summary."""
    print("=" * 70)
    print("Optuna Trial 分析摘要")
    print("=" * 70)

    n_total = len(trials)
    n_complete = sum(1 for t in trials if t.get("state") == "COMPLETE")
    n_pruned = sum(1 for t in trials if t.get("state") == "PRUNED")
    n_fail = sum(1 for t in trials if t.get("state") == "FAIL")

    print(f"\nTrial统计:")
    print(f"  总计: {n_total}")
    print(f"  成功: {n_complete}")
    print(f"  剪枝: {n_pruned}")
    print(f"  失败: {n_fail}")

    if n_complete == 0:
        print("\n❌ 无成功trial")
        return

    # Score distribution
    scores = [t["value"] for t in trials if t.get("state") == "COMPLETE"]
    print(f"\nScore分布:")
    print(f"  均值: {np.mean(scores):.6f}")
    print(f"  标准差: {np.std(scores):.6f}")
    print(f"  最小: {np.min(scores):.6f}")
    print(f"  最大: {np.max(scores):.6f}")
    print(f"  中位数: {np.median(scores):.6f}")

    # Factor analysis
    best_config = load_best_config()
    factor_probs = best_config.get("factor_probs", {})
    n_factors = len(factor_probs)

    print(f"\n因子分析 (共{n_factors}个候选):")
    factor_freq, n_completed = analyze_factor_frequency(trials, n_factors)

    # Load factor names from candidate pool
    report_file = project_root / "factor_candidate_pool.json"
    factor_names = []
    if report_file.exists():
        report = json.loads(report_file.read_text())
        for fc in report.get("all_computed_factors", []):
            factor_names.append(fc["name"])

    print(f"  {'因子索引':>6s}  {'因子名称':45s}  {'选择次数':>8s}  {'频率':>8s}  {'最优选中':>8s}")
    print(f"  {'-'*85}")
    for idx, freq_info in sorted(factor_freq.items(), key=lambda x: -x[1]["frequency"]):
        name = factor_names[idx] if idx < len(factor_names) else f"factor_{idx}"
        best_selected = "✅" if best_config.get("factor_mask", {}).get(f"factor_prob_{idx}", 0) == 1 else "❌"
        print(
            f"  {idx:>6d}  {name:45s}  {freq_info['count']:>8d}  "
            f"{freq_info['frequency']:>7.1%}  {best_selected:>8s}"
        )

    # Parameter importance
    print(f"\n参数重要性 (基于与score的相关性):")
    importance = analyze_param_importance(trials)
    for param, imp in list(importance.items())[:15]:
        bar = "█" * int(imp * 20)
        print(f"  {param:25s}  {imp:.3f}  {bar}")

    # Convergence
    print(f"\n收敛分析:")
    convergence = analyze_convergence(trials)
    print(f"  最终最优Score: {convergence['final_best']:.6f}")
    print(f"  最优Trial #: {convergence['best_trial']['trial'] if convergence['best_trial'] else 'N/A'}")
    if convergence['trajectory']:
        traj = convergence['trajectory']
        print(f"  最后{len(traj)}个trial收敛曲线:")
        print(f"    Trial#  BestSoFar")
        for point in traj[-10:]:
            print(f"    {point['trial']:>6d}  {point['best_so_far']:.6f}")

    # LGB parameter distribution for top 20% trials
    print(f"\n最优20% Trial的LGB参数分布:")
    threshold = np.percentile(scores, 80)
    top_trials = [t for t in trials if t.get("state") == "COMPLETE" and t["value"] >= threshold]

    lgb_params = {}
    for t in top_trials:
        for k, v in t.get("params", {}).items():
            if not k.startswith("factor_prob_"):
                if k not in lgb_params:
                    lgb_params[k] = []
                lgb_params[k].append(v)

    for param, values in sorted(lgb_params.items()):
        if not values:
            continue
        arr = np.array(values, dtype=float)
        print(f"  {param:25s}  mean={np.mean(arr):.4f}  std={np.std(arr):.4f}  "
              f"min={np.min(arr):.4f}  max={np.max(arr):.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Optuna Trial分析")
    parser.add_argument("--plot-importance", action="store_true", help="参数重要性图")
    parser.add_argument("--plot-convergence", action="store_true", help="收敛曲线")
    parser.add_argument("--factor-frequency", action="store_true", help="因子选择频率")
    args = parser.parse_args()

    trials = load_trial_results()
    best_config = load_best_config()

    # Default: print full summary
    if not any([args.plot_importance, args.plot_convergence, args.factor_frequency]):
        print_summary(trials)
        return 0

    # Factor frequency
    if args.factor_frequency:
        factor_probs = best_config.get("factor_probs", {})
        n_factors = len(factor_probs)
        factor_freq, n_completed = analyze_factor_frequency(trials, n_factors)

        report_file = project_root / "factor_candidate_pool.json"
        factor_names = []
        if report_file.exists():
            report = json.loads(report_file.read_text())
            for fc in report.get("all_computed_factors", []):
                factor_names.append(fc["name"])

        print("因子选择频率:")
        for idx, freq_info in sorted(factor_freq.items(), key=lambda x: -x[1]["frequency"]):
            name = factor_names[idx] if idx < len(factor_names) else f"factor_{idx}"
            bar = "█" * int(freq_info["frequency"] * 30)
            print(f"  {name:45s}  {freq_info['frequency']:>7.1%}  {bar}")

    # Parameter importance
    if args.plot_importance:
        importance = analyze_param_importance(trials)
        print("参数重要性:")
        for param, imp in list(importance.items())[:20]:
            bar = "█" * int(imp * 30)
            print(f"  {param:25s}  {imp:.3f}  {bar}")

    # Convergence
    if args.plot_convergence:
        convergence = analyze_convergence(trials)
        print("收敛轨迹 (最后20个trial):")
        for point in convergence["trajectory"]:
            bar = "█" * int((point["best_so_far"] + 0.1) * 200)  # rough scaling
            print(f"  Trial {point['trial']:>4d}  {point['best_so_far']:.6f}  {bar}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
