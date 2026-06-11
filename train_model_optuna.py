#!/usr/bin/env python3
"""
Phase 2: Optuna联合搜索 — QuantaAlpha v2.0

用Optuna TPE算法同时搜索最优因子组合和LightGBM超参数。

搜索策略：
  - 因子选择：连续松弛（factor_prob ~ Uniform(0,1), mask = prob >= 0.5）
  - LightGBM超参数：log-uniform/float/int范围搜索
  - 目标函数：mean_ic + ICIR门控奖励/惩罚 + 回撤惩罚 + IC门控惩罚

使用方法：
    source venv/bin/activate
    python3 train_model_optuna.py              # 默认200 trials
    python3 train_model_optuna.py --n-trials 100  # 自定义trial数
    python3 train_model_optuna.py --n-trials 50 --seed 42  # 固定随机种子
"""

import json
import sys
import time
import os
import logging
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "optuna_search.log", mode='a'),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================
DEFAULT_N_TRIALS = 200
RANDOM_SEED = 42
MIN_FACTORS = 3
MAX_FACTORS = 15

# Quality gate thresholds (matching v2.0 standard)
IC_THRESHOLD = 0.015
ICIR_THRESHOLD = 0.12
MAX_DD_THRESHOLD = 0.15
IC_POSITIVE_RATIO_THRESHOLD = 0.55
WIN_RATE_THRESHOLD = 0.52
ANN_RETURN_THRESHOLD = 0.05

# Time split: train 2016-2023, valid 2024, test 2025
TRAIN_END_DATE = pd.Timestamp("2023-12-31")
VALID_START_DATE = pd.Timestamp("2024-01-01")
VALID_END_DATE = pd.Timestamp("2024-12-31")
TEST_START_DATE = pd.Timestamp("2025-01-01")


# ============================================================
# 数据加载
# ============================================================
def load_feature_cache() -> tuple:
    """Load cached features from HDF5.

    Returns (data_df, factor_cols, factor_names_from_report).
    """
    labeled_file = project_root / "feature_cache_labeled.h5"
    if not labeled_file.exists():
        print(f"❌ 特征缓存不存在: {labeled_file}")
        print("请先运行: python3 compute_feature_cache.py")
        sys.exit(1)

    print(f"加载特征缓存: {labeled_file}")
    data_df = pd.read_hdf(labeled_file, key='data')
    print(f"  数据量: {data_df.shape}")

    # Load factor candidate pool report for names
    report_file = project_root / "factor_candidate_pool.json"
    factor_names = []
    if report_file.exists():
        report = json.loads(report_file.read_text())
        for fc in report.get("all_computed_factors", []):
            factor_names.append(fc["name"])
        print(f"  因子名称: {len(factor_names)} 个（来自报告）")
    else:
        # Fallback: just use column names
        print("  ⚠️  无候选池报告，使用默认因子名")

    factor_cols = [c for c in data_df.columns if c.startswith('factor_')]
    if not factor_cols:
        # Maybe derived features only
        factor_cols = [c for c in data_df.columns if c not in ['label']]

    if 'label' not in data_df.columns:
        print("❌ 缓存中无label列，请确认使用feature_cache_labeled.h5")
        sys.exit(1)

    return data_df, factor_cols, factor_names


# ============================================================
# 数据切分
# ============================================================
def split_data(data_df: pd.DataFrame, factor_cols: list) -> dict:
    """Split data into train/valid/test sets by date."""
    dates_index = data_df.index.get_level_values(1)

    train_mask = dates_index <= TRAIN_END_DATE
    valid_mask = (dates_index >= VALID_START_DATE) & (dates_index <= VALID_END_DATE)
    test_mask = dates_index >= TEST_START_DATE

    train_df = data_df[train_mask].dropna(subset=factor_cols + ['label'])
    valid_df = data_df[valid_mask].dropna(subset=factor_cols + ['label'])
    test_df = data_df[test_mask].dropna(subset=factor_cols + ['label'])

    # Fallback if valid is empty
    if len(valid_df) == 0:
        logger.warning("验证集为空，使用训练集最后20%作为验证")
        all_dates = sorted(dates_index.unique())
        split_idx = int(len(all_dates) * 0.80)
        train_df = data_df[dates_index < all_dates[split_idx]].dropna(subset=factor_cols + ['label'])
        valid_df = data_df[dates_index >= all_dates[split_idx]].dropna(subset=factor_cols + ['label'])
        test_df = valid_df

    print(f"  训练集: {len(train_df)} ({str(TRAIN_END_DATE)[:10]}前)")
    print(f"  验证集: {len(valid_df)} (2024)")
    print(f"  测试集: {len(test_df)} (2025)")

    return {
        'train_df': train_df,
        'valid_df': valid_df,
        'test_df': test_df,
    }


# ============================================================
# 评估函数
# ============================================================
def compute_daily_ics(y_pred: np.ndarray, y_true: np.ndarray,
                      dates: pd.Index) -> list:
    """Compute daily cross-sectional IC."""
    daily_ics = []
    for date in dates.unique():
        mask = dates == date
        if mask.sum() < 5:
            continue
        try:
            ic = np.corrcoef(y_pred[mask], y_true[mask])[0, 1]
            if not np.isnan(ic):
                daily_ics.append(ic)
        except Exception:
            continue
    return daily_ics


def evaluate_model(model, X: np.ndarray, y: np.ndarray,
                   dates: pd.Index) -> dict:
    """Evaluate model predictions with full metrics."""
    y_pred = model.predict(X)

    daily_ics = compute_daily_ics(y_pred, y, dates)

    mean_ic = np.mean(daily_ics) if daily_ics else 0
    ic_std = np.std(daily_ics) if daily_ics else 1e-8
    icir = mean_ic / ic_std if ic_std > 0 else 0
    ic_positive_ratio = sum(1 for ic in daily_ics if ic > 0) / len(daily_ics) if daily_ics else 0

    # Portfolio metrics: date-grouped top-K long-short returns
    df_tmp = pd.DataFrame({
        'pred': y_pred,
        'label': y,
        'date': dates,
    })

    # Daily top-K long-short returns (Top 20% - Bottom 20%)
    daily_ls_returns = []
    daily_correct_sign = []
    for date, grp in df_tmp.groupby('date'):
        if len(grp) < 10:
            continue
        n_top = max(1, len(grp) // 5)
        sorted_grp = grp.sort_values('pred')
        top_ret = sorted_grp['label'].iloc[-n_top:].mean()
        bottom_ret = sorted_grp['label'].iloc[:n_top].mean()
        daily_ls_returns.append(top_ret - bottom_ret)
        # Sign accuracy
        daily_correct_sign.append(1 if (top_ret > bottom_ret) == (grp['pred'].corr(grp['label']) > 0) else 0)

    if daily_ls_returns:
        ls_returns_arr = np.array(daily_ls_returns)
        win_rate = np.mean([1 for r in daily_ls_returns if r > 0])
        # Cumulative return and max drawdown
        cum_returns = np.cumprod(1 + ls_returns_arr) - 1
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / (1 + running_max)
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        long_short_return = float(np.mean(daily_ls_returns)) * 252  # annualized
    else:
        win_rate = 0.5
        max_dd = 0
        long_short_return = 0

    mse = np.mean((y_pred - y) ** 2)

    return {
        'mean_ic': float(mean_ic),
        'ic_std': float(ic_std),
        'icir': float(icir),
        'ic_positive_ratio': float(ic_positive_ratio),
        'mse': float(mse),
        'win_rate': float(win_rate),
        'long_short_return': float(long_short_return),
        'max_drawdown': float(max_dd),
        'n_daily_ics': len(daily_ics),
    }


# ============================================================
# 目标函数
# ============================================================
def compute_score(metrics: dict) -> float:
    """
    Objective function score.

    score = mean_ic                          # 基础分数
          + 0.005 * I(ICIR > 0.12)           # ICIR通过门控奖励
          - 0.01 * max(0, 0.12 - ICIR)       # ICIR未通过惩罚
          - 0.02 * max(0, |max_dd| - 0.15)   # 回撤超标惩罚
          - 0.05 * I(mean_ic < 0.015)        # IC门控失败重罚
          - 0.02 * I(IC_positive_ratio < 0.55)  # IC>0比例未通过惩罚
    """
    mean_ic = metrics['mean_ic']
    icir = metrics['icir']
    max_dd = abs(metrics['max_drawdown'])
    ic_positive_ratio = metrics['ic_positive_ratio']

    score = mean_ic

    # ICIR bonus/penalty
    if icir > ICIR_THRESHOLD:
        score += 0.005
    else:
        score -= 0.01 * max(0, ICIR_THRESHOLD - icir)

    # Drawdown penalty
    if max_dd > MAX_DD_THRESHOLD:
        score -= 0.02 * (max_dd - MAX_DD_THRESHOLD)

    # IC gate failure penalty
    if mean_ic < IC_THRESHOLD:
        score -= 0.05

    # IC positive ratio penalty
    if ic_positive_ratio < IC_POSITIVE_RATIO_THRESHOLD:
        score -= 0.02

    return score


# ============================================================
# Optuna objective
# ============================================================
def create_objective(data_splits: dict, factor_cols: list, factor_names: list):
    """Create Optuna objective function."""

    train_df = data_splits['train_df']
    valid_df = data_splits['valid_df']

    dates_train = train_df.index.get_level_values(1)
    dates_valid = valid_df.index.get_level_values(1)

    y_train_all = train_df['label'].values
    y_valid_all = valid_df['label'].values

    n_factors = len(factor_cols)

    def objective(trial):
        import lightgbm as lgb

        # --- Factor selection (continuous relaxation) ---
        factor_probs = []
        for i in range(n_factors):
            prob = trial.suggest_float(f"factor_prob_{i}", 0.0, 1.0)
            factor_probs.append(prob)

        # Threshold to binary mask
        mask = [1 if p >= 0.5 else 0 for p in factor_probs]
        n_selected = sum(mask)

        # Constraint: 3 <= n_selected <= 15
        if n_selected < MIN_FACTORS or n_selected > MAX_FACTORS:
            # Penalty: return a very bad score
            trial.set_user_attr("n_selected", n_selected)
            trial.set_user_attr("status", "constraint_violated")
            return -1.0

        selected_cols = [factor_cols[i] for i in range(n_factors) if mask[i]]

        X_train = train_df[selected_cols].values
        X_valid = valid_df[selected_cols].values

        # Check for all-NaN or constant features
        if np.all(np.isnan(X_train)) or np.all(np.isnan(X_valid)):
            return -1.0

        # Fill NaN with 0 (shouldn't happen after dropna, but be safe)
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_valid = np.nan_to_num(X_valid, nan=0.0)

        # --- LightGBM hyperparameters ---
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 500),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 500),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'num_threads': 20,
            'verbose': -1,
            'seed': RANDOM_SEED,
        }

        # --- Train LightGBM ---
        try:
            train_data = lgb.Dataset(X_train, label=y_train_all)
            valid_data = lgb.Dataset(X_valid, label=y_valid_all, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[valid_data],
                valid_names=['valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0),  # silent
                ]
            )
        except Exception as e:
            logger.warning(f"训练失败: {e}")
            return -1.0

        # --- Evaluate on validation set ---
        metrics = evaluate_model(model, X_valid, y_valid_all, dates_valid)

        # --- Compute score ---
        score = compute_score(metrics)

        # --- Store trial info ---
        selected_names = [factor_names[i] for i in range(n_factors) if mask[i]]
        trial.set_user_attr("n_selected", n_selected)
        trial.set_user_attr("selected_factors", selected_names)
        trial.set_user_attr("status", "completed")
        trial.set_user_attr("score", score)
        for k, v in metrics.items():
            trial.set_user_attr(k, v)

        return score

    return objective


# ============================================================
# 运行搜索
# ============================================================
def run_search(n_trials: int = DEFAULT_N_TRIALS, seed: int = RANDOM_SEED):
    """Run Optuna TPE search."""

    # Load data
    print("\n加载特征缓存...")
    data_df, factor_cols, factor_names_raw = load_feature_cache()

    # Ensure we have factor names (fallback if report didn't have all)
    if len(factor_names_raw) < len(factor_cols):
        factor_names = [f"factor_{i+1}" for i in range(len(factor_cols))]
    else:
        factor_names = factor_names_raw[:len(factor_cols)]

    print(f"  因子数: {len(factor_cols)}")
    print(f"  样本数: {len(data_df)}")

    # Split data
    print("\n切分数据...")
    data_splits = split_data(data_df, factor_cols)

    # Create objective
    objective = create_objective(data_splits, factor_cols, factor_names)

    # Create Optuna study
    results_dir = project_root / "optuna_search_results"
    results_dir.mkdir(exist_ok=True)

    db_path = results_dir / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"

    study_name = "quantaalpha_v2_optuna_search"

    # Try to load existing study or create new one
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        n_existing = len(study.trials)
        print(f"\n加载已有研究: {n_existing} trials 已完成")
        remaining = max(0, n_trials - n_existing)
        if remaining == 0:
            print(f"已达到目标trial数 ({n_trials})，跳过搜索")
        else:
            print(f"继续搜索 {remaining} 个额外trials")
    except Exception:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        remaining = n_trials
        print(f"\n创建新研究: {n_trials} trials")

    if remaining <= 0:
        # Still save results
        save_results(study, factor_cols, factor_names)
        return study

    # Run optimization
    print(f"\n{'='*70}")
    print(f"Optuna联合搜索开始 ({remaining} trials)")
    print(f"{'='*70}")

    start_time = time.time()

    # Callback for progress logging
    def logging_callback(study, trial):
        if trial.number % 10 == 0 or trial.number == remaining - 1:
            elapsed = time.time() - start_time
            avg_time = elapsed / (trial.number + 1)
            est_remaining = avg_time * (remaining - trial.number - 1)

            logger.info(
                f"Trial {trial.number}: "
                f"score={trial.value:.4f}, "
                f"IC={study.trials[trial.number].user_attrs.get('mean_ic', 0):.4f}, "
                f"ICIR={study.trials[trial.number].user_attrs.get('icir', 0):.4f}, "
                f"factors={study.trials[trial.number].user_attrs.get('n_selected', 0)}, "
                f"elapsed={elapsed:.0f}s, "
                f"est_remaining={est_remaining:.0f}s"
            )

            # Auto-save intermediate results every 10 trials
            if trial.number % 10 == 0:
                try:
                    save_intermediate_results(study, factor_cols, factor_names)
                except Exception as e:
                    logger.warning(f"中间结果保存失败: {e}")

    try:
        study.optimize(objective, n_trials=remaining, callbacks=[logging_callback],
                       n_jobs=1, gc_after_trial=True)
    except KeyboardInterrupt:
        logger.warning("搜索被中断，保存中间结果...")
    except Exception as e:
        logger.error(f"搜索异常: {e}")

    elapsed = time.time() - start_time
    print(f"\n搜索完成！耗时: {elapsed:.0f}s ({elapsed/60:.1f}分钟)")

    # Save final results
    save_results(study, factor_cols, factor_names)

    return study


# ============================================================
# 结果保存
# ============================================================
def save_intermediate_results(study, factor_cols: list, factor_names: list):
    """Save intermediate results."""
    results_dir = project_root / "optuna_search_results"

    if len(study.trials) == 0:
        return

    # Save best trial info
    try:
        best = study.best_trial
        best_config = {
            "trial_number": best.number,
            "score": best.value,
            "factor_params": {},
            "lgb_params": {},
            "metrics": {},
        }

        for k, v in best.params.items():
            if k.startswith("factor_prob_"):
                best_config["factor_params"][k] = v
            else:
                best_config["lgb_params"][k] = v

        for k, v in best.user_attrs.items():
            if isinstance(v, (int, float, str, bool, list, dict)):
                best_config["metrics"][k] = v

        with open(results_dir / "best_intermediate.json", 'w') as f:
            json.dump(best_config, f, indent=2, default=str)
    except Exception:
        pass


def save_results(study, factor_cols: list, factor_names: list):
    """Save complete results."""
    results_dir = project_root / "optuna_search_results"
    results_dir.mkdir(exist_ok=True)

    n_trials = len(study.trials)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print(f"\n{'='*70}")
    print(f"搜索结果: {len(completed_trials)}/{n_trials} trials 成功")
    print(f"{'='*70}")

    if not completed_trials:
        print("❌ 无成功trial")
        return

    # Best trial
    best = study.best_trial
    print(f"\n最优Trial #{best.number}:")
    print(f"  Score: {best.value:.6f}")

    # Parse best config
    factor_probs = {k: v for k, v in best.params.items() if k.startswith("factor_prob_")}
    factor_mask = {k: (1 if v >= 0.5 else 0) for k, v in factor_probs.items()}
    n_selected = sum(factor_mask.values())

    selected_idx = [i for i in range(len(factor_cols)) if factor_mask.get(f"factor_prob_{i}", 0) == 1]
    selected_names = [factor_names[i] for i in selected_idx]
    selected_cols = [factor_cols[i] for i in selected_idx]

    lgb_params = {k: v for k, v in best.params.items() if not k.startswith("factor_prob_")}

    print(f"  因子数: {n_selected}")
    print(f"  选中因子: {selected_names}")
    print(f"  LGB参数:")
    for k, v in lgb_params.items():
        print(f"    {k}: {v}")

    # Print metrics
    for k, v in best.user_attrs.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Save best configuration
    # Load factor expressions from candidate pool report
    report_file = project_root / "factor_candidate_pool.json"
    selected_expressions = []
    if report_file.exists():
        report = json.loads(report_file.read_text())
        expr_map = {}
        for fc in report.get("all_computed_factors", []):
            expr_map[fc["name"]] = fc.get("expression", fc.get("qlib_expression", ""))
        for name in selected_names:
            expr = expr_map.get(name, "")
            if not expr:
                # Fallback: use column name as placeholder
                expr = f"factor_{selected_names.index(name)+1}"
            selected_expressions.append(expr)

    best_config = {
        "trial_number": best.number,
        "score": best.value,
        "n_factors": int(n_selected),
        "selected_factors": selected_names,
        "selected_columns": selected_cols,
        "selected_expressions": selected_expressions,
        "factor_probs": {k: round(v, 6) for k, v in factor_probs.items()},
        "factor_mask": {k: int(v) for k, v in factor_mask.items()},
        "lgb_params": {k: round(v, 6) if isinstance(v, float) else v for k, v in lgb_params.items()},
        "metrics": {k: v for k, v in best.user_attrs.items() if isinstance(v, (int, float, str, bool))},
        "timestamp": datetime.datetime.now().isoformat(),
    }

    with open(results_dir / "best_configuration.json", 'w') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 最优配置: {results_dir / 'best_configuration.json'}")

    # Save all trials
    all_trials = []
    for t in completed_trials:
        trial_info = {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "user_attrs": {k: v for k, v in t.user_attrs.items()
                          if isinstance(v, (int, float, str, bool, list))},
        }
        all_trials.append(trial_info)

    all_trials.sort(key=lambda x: x["value"], reverse=True)

    with open(results_dir / "all_trials.json", 'w') as f:
        json.dump(all_trials, f, indent=2, default=str)
    print(f"✅ 全部trials: {results_dir / 'all_trials.json'}")

    # Print top 10 trials
    print(f"\nTop 10 Trials:")
    print(f"  {'Trial#':>6s}  {'Score':>10s}  {'IC':>8s}  {'ICIR':>8s}  {'Factors':>7s}  {'MaxDD':>8s}")
    print(f"  {'-'*55}")
    for t_info in all_trials[:10]:
        attrs = t_info.get("user_attrs", {})
        ic_val = attrs.get("mean_ic", 0)
        icir_val = attrs.get("icir", 0)
        n_f = attrs.get("n_selected", 0)
        mdd = attrs.get("max_drawdown", 0)
        print(
            f"  {t_info['number']:>6d}  {t_info['value']:>10.4f}  "
            f"{ic_val:>8.4f}  {icir_val:>8.4f}  {n_f:>7d}  {mdd:>8.4f}"
        )


# ============================================================
# 主流程
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Optuna联合搜索")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                        help=f"Trial数量（默认{DEFAULT_N_TRIALS}）")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"随机种子（默认{RANDOM_SEED}）")
    parser.add_argument("--resume", action="store_true",
                        help="继续上次未完成的研究")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 2: Optuna联合搜索 (QuantaAlpha v2.0)")
    print("=" * 70)
    print(f"  Trial数: {args.n_trials}")
    print(f"  随机种子: {args.seed}")
    print(f"  因子约束: {MIN_FACTORS} ~ {MAX_FACTORS}")
    print(f"  算法: TPE (Tree-structured Parzen Estimator)")

    run_search(n_trials=args.n_trials, seed=args.seed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
