#!/usr/bin/env python3
"""
综合评估脚本 — QuantaAlpha模型完整评测

功能：
1. 加载已有模型 + 因子表达式
2. 计算完整评测指标（IC/ICIR/回撤/胜率/夏普/年化）
3. 分年度评估（检测时间衰减）
4. 分regime评估（牛/熊/震荡）
5. 输出综合评分报告 + 是否通过质量门控

使用方法：
    source venv/bin/activate
    # 评估v1.0模型（默认三因子）
    python3 evaluate_model_comprehensive.py

    # 指定模型文件和因子表达式
    python3 evaluate_model_comprehensive.py \
        --model-file exported_models/quantaalpha_model.txt \
        --factors "expr1,expr2,expr3"

    # 指定测试集时间范围
    python3 evaluate_model_comprehensive.py --test-start 2025-01-01 --test-end 2025-12-26
"""

import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

_script_dir = Path(__file__).resolve().parent
# Support packaged layout: when running from algorithm/, resolve to package root
if (_script_dir.parent / "models").exists() and (_script_dir.parent / "docs").exists():
    project_root = _script_dir.parent  # packaged: algorithm/ → parent
elif (_script_dir / "models").exists():
    project_root = _script_dir  # already at package root
else:
    project_root = _script_dir  # source tree
sys.path.insert(0, str(project_root))


# ============================================================
# 质量门控阈值（v2.0标准）
# ============================================================
QUALITY_GATES = {
    "IC": 0.015,
    "ICIR": 0.12,
    "AnnReturn": 0.05,
    "MaxDrawdown": -0.15,  # 负值表示最大回撤上限
    "WinRate": 0.52,
    "IC_Positive_Ratio": 0.55,
}


def parse_args():
    # Detect package root for both source-tree and packaged layouts
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # If running from algorithm/ inside a package, resolve to package root (parent)
    if os.path.isdir(os.path.join(script_dir, "models")):
        # Running from package root directly
        package_root = script_dir
        default_model = "models/quantaalpha_model.txt"
    elif os.path.isdir(os.path.join(os.path.dirname(script_dir), "models")):
        # Running from algorithm/ inside package
        package_root = os.path.dirname(script_dir)
        default_model = "models/quantaalpha_model.txt"
    elif os.path.isdir(os.path.join(script_dir, "exported_models")):
        # Source tree
        package_root = script_dir
        default_model = "exported_models/quantaalpha_model.txt"
    else:
        # Fallback: source tree
        package_root = script_dir
        default_model = "exported_models/quantaalpha_model.txt"

    parser = argparse.ArgumentParser(description="QuantaAlpha模型综合评估")
    parser.add_argument("--model-file", type=str,
                        default=default_model,
                        help="模型文件路径（.txt或.pkl）")
    parser.add_argument("--factors", type=str, default=None,
                        help="因子表达式，逗号分隔。默认使用v1.0三因子")
    parser.add_argument("--train-start", type=str, default="2016-01-01",
                        help="训练集开始日期")
    parser.add_argument("--train-end", type=str, default="2023-12-31",
                        help="训练集结束日期")
    parser.add_argument("--valid-start", type=str, default="2024-01-01",
                        help="验证集开始日期")
    parser.add_argument("--valid-end", type=str, default="2024-12-31",
                        help="验证集结束日期")
    parser.add_argument("--test-start", type=str, default="2025-01-01",
                        help="测试集开始日期")
    parser.add_argument("--test-end", type=str, default="2025-12-26",
                        help="测试集结束日期")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K组合选股数")
    parser.add_argument("--no-gate", action="store_true",
                        help="跳过质量门控检查（仅输出指标）")
    return parser.parse_args()


# ============================================================
# 数据加载
# ============================================================
def load_data(factor_expressions, start_time, end_time):
    """使用Qlib加载特征和标签数据"""
    import qlib
    from qlib.data import D

    provider_uri = Path.home() / ".qlib/qlib_data/cn_data"
    qlib.init(provider_uri=str(provider_uri), region='cn')

    instruments_obj = D.instruments('csi300')
    instruments = D.list_instruments(
        instruments=instruments_obj,
        start_time=start_time,
        end_time=end_time,
        as_list=True,
    )

    print(f"  标的: CSI300 ({len(instruments)} stocks)")
    print(f"  时间: {start_time} ~ {end_time}")

    # 计算特征
    features_df = D.features(instruments, factor_expressions,
                             start_time=start_time, end_time=end_time)
    features_df.columns = [f'factor_{i+1}' for i in range(len(factor_expressions))]

    # 计算标签（未来2日收益率，与train_model_simple.py一致）
    label_expr = "Ref($close, -2)/$close - 1"
    label_df = D.features(instruments, [label_expr],
                          start_time=start_time, end_time=end_time)
    label_df.columns = ['label']

    # 合并清理
    data_df = features_df.join(label_df, how='inner').dropna()

    print(f"  有效样本: {len(data_df)}")
    print(f"  NaN比例: {features_df.isnull().sum().sum() / features_df.size * 100:.2f}%")

    return data_df


def split_data(data_df, train_start, train_end, valid_start, valid_end, test_start, test_end):
    """按日期切分训练/验证/测试集"""
    # 将字符串日期转为Timestamp
    train_end_ts = pd.Timestamp(train_end)
    valid_start_ts = pd.Timestamp(valid_start)
    valid_end_ts = pd.Timestamp(valid_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    dates = data_df.index.get_level_values(1)

    train_mask = (dates >= pd.Timestamp(train_start)) & (dates <= train_end_ts)
    valid_mask = (dates >= valid_start_ts) & (dates <= valid_end_ts)
    test_mask = (dates >= test_start_ts) & (dates <= test_end_ts)

    train_df = data_df[train_mask]
    valid_df = data_df[valid_mask]
    test_df = data_df[test_mask]

    print(f"  训练集: {len(train_df)} ({train_start} ~ {train_end})")
    print(f"  验证集: {len(valid_df)} ({valid_start} ~ {valid_end})")
    print(f"  测试集: {len(test_df)} ({test_start} ~ {test_end})")

    return train_df, valid_df, test_df


# ============================================================
# 模型加载
# ============================================================
def load_model(model_path):
    """加载LightGBM模型"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    if str(model_path).endswith('.txt'):
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
        print(f"  模型格式: LightGBM .txt ({model_path.stat().st_size / 1024:.1f} KB)")
    elif str(model_path).endswith('.pkl'):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"  模型格式: Pickle ({model_path.stat().st_size / 1024:.1f} KB)")
    else:
        raise ValueError(f"不支持的模型格式: {model_path}")

    print(f"  树数量: {model.num_trees()}")
    return model


# ============================================================
# 核心指标计算
# ============================================================
def compute_ic_metrics(y_pred, y_true):
    """计算IC/ICIR"""
    ic = np.corrcoef(y_pred, y_true)[0, 1]
    ic_std = np.std(y_pred) * np.std(y_true)
    if ic_std < 1e-12:
        return ic, 0.0
    icir = ic / (np.std(y_pred) * np.std(y_true) / np.sqrt(len(y_true))) if np.std(y_true) > 1e-12 else 0.0
    # 简化ICIR：IC / IC_std 近似
    icir = ic / (abs(ic) + 1e-12) * np.sqrt(len(y_true)) / 100 if False else ic / (np.std(y_true) + 1e-12)

    # 更准确的ICIR定义：IC均值 / IC标准差（用于时序）
    # 这里用单期IC，ICIR = IC / prediction_std近似
    icir_simple = ic / (np.std(y_pred) + 1e-12) * np.sqrt(len(y_true))

    return ic, icir_simple


def compute_daily_ic(y_pred, y_true, dates):
    """按日期分组计算每日IC"""
    df = pd.DataFrame({
        'pred': y_pred,
        'true': y_true,
        'date': dates
    })

    daily_ics = []
    for date, group in df.groupby('date'):
        if len(group) < 10:
            continue
        ic = group['pred'].corr(group['true'])
        if not np.isnan(ic):
            daily_ics.append({'date': date, 'ic': ic})

    ic_series = pd.DataFrame(daily_ics)
    if len(ic_series) == 0:
        return {}

    mean_ic = ic_series['ic'].mean()
    std_ic = ic_series['ic'].std()
    icir = mean_ic / (std_ic + 1e-12)
    ic_positive_ratio = (ic_series['ic'] > 0).mean()

    return {
        'Mean_IC': mean_ic,
        'IC_Std': std_ic,
        'ICIR': icir,
        'IC_Positive_Ratio': ic_positive_ratio,
        'IC_Series': ic_series,
    }


def compute_portfolio_metrics(y_pred, y_true, dates, top_k=50):
    """
    基于预测排序的Top-K组合模拟
    计算：年化收益、最大回撤、胜率、夏普比率
    """
    df = pd.DataFrame({
        'pred': y_pred,
        'true': y_true,
        'date': dates
    })

    daily_returns = []
    for date, group in df.groupby('date'):
        if len(group) < top_k:
            continue
        # 选Top-K
        top_k_stocks = group.nlargest(top_k, 'pred')
        # 等权组合收益
        daily_ret = top_k_stocks['true'].mean()
        daily_returns.append({'date': date, 'return': daily_ret})

    ret_series = pd.DataFrame(daily_returns).set_index('date')['return']
    if len(ret_series) == 0:
        return {}

    # 累计收益曲线
    cum_ret = (1 + ret_series).cumprod() - 1

    # 年化收益
    n_days = len(ret_series)
    total_ret = (1 + ret_series).prod() - 1
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 else 0

    # 最大回撤
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / (rolling_max + 1)
    max_drawdown = drawdown.min()

    # 夏普比率（日频）
    sharpe = ret_series.mean() / (ret_series.std() + 1e-12) * np.sqrt(252)

    # 胜率（日度正收益比例）
    win_rate = (ret_series > 0).mean()

    return {
        'AnnReturn': ann_ret,
        'TotalReturn': total_ret,
        'MaxDrawdown': max_drawdown,
        'Sharpe': sharpe,
        'WinRate': win_rate,
        'N_TradingDays': n_days,
    }


# ============================================================
# 分年度评估
# ============================================================
def compute_yearly_breakdown(y_pred, y_true, dates, top_k=50):
    """分年度计算IC和组合收益"""
    df = pd.DataFrame({
        'pred': y_pred,
        'true': y_true,
        'date': dates
    })
    df['year'] = pd.to_datetime(df['date']).dt.year

    yearly_results = {}
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        if len(year_data) < 100:
            continue

        ic = year_data['pred'].corr(year_data['true'])
        year_daily_ics = []
        for date, group in year_data.groupby('date'):
            if len(group) >= 10:
                ic_d = group['pred'].corr(group['true'])
                if not np.isnan(ic_d):
                    year_daily_ics.append(ic_d)

        yearly_results[str(year)] = {
            'IC': ic if not np.isnan(ic) else 0,
            'N_Days': len(year_daily_ics),
            'IC_Positive_Ratio': np.mean([x > 0 for x in year_daily_ics]) if year_daily_ics else 0,
        }

    return yearly_results


# ============================================================
# 分Regime评估
# ============================================================
def classify_regime(year):
    """
    简单regime分类：基于沪深300年涨跌幅
    >15% = 牛市, <-15% = 熊市, 其他 = 震荡市

    历史参考：
    2016: -11%  震荡
    2017: +22%  牛市
    2018: -25%  熊市
    2019: +36%  牛市
    2020: +27%  牛市
    2021: -5%   震荡
    2022: -21%  熊市
    2023: -11%  震荡
    2024: +15%  牛市（预估）
    2025: TBD
    """
    # 简化：直接基于年份映射
    regime_map = {
        2016: "Sideways",
        2017: "Bull",
        2018: "Bear",
        2019: "Bull",
        2020: "Bull",
        2021: "Sideways",
        2022: "Bear",
        2023: "Sideways",
        2024: "Bull",
        2025: "Unknown",
    }
    return regime_map.get(year, "Unknown")


def compute_regime_metrics(y_pred, y_true, dates, top_k=50):
    """分regime计算指标"""
    df = pd.DataFrame({
        'pred': y_pred,
        'true': y_true,
        'date': dates
    })
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['regime'] = df['year'].map(lambda y: classify_regime(y))

    regime_results = {}
    for regime in ["Bull", "Bear", "Sideways"]:
        regime_data = df[df['regime'] == regime]
        if len(regime_data) < 100:
            regime_results[regime] = {"IC": None, "N_Samples": len(regime_data), "Note": "样本不足"}
            continue

        ic = regime_data['pred'].corr(regime_data['true'])

        # 简单组合收益
        daily_rets = []
        for date, group in regime_data.groupby('date'):
            if len(group) >= top_k:
                daily_rets.append(group.nlargest(top_k, 'pred')['true'].mean())

        if daily_rets:
            ann_ret = np.prod(1 + np.array(daily_rets)) ** (252 / len(daily_rets)) - 1
            win_rate = np.mean([r > 0 for r in daily_rets])
        else:
            ann_ret = None
            win_rate = None

        regime_results[regime] = {
            'IC': ic if not np.isnan(ic) else 0,
            'AnnReturn': ann_ret,
            'WinRate': win_rate,
            'N_Samples': len(regime_data),
            'N_TradingDays': len(daily_rets),
        }

    return regime_results


# ============================================================
# 质量门控检查
# ============================================================
def check_quality_gates(metrics, gates=None):
    """检查是否通过质量门控"""
    gates = gates or QUALITY_GATES
    results = {}
    all_pass = True

    gate_checks = {
        "IC": lambda m: m.get("Mean_IC", 0) >= gates["IC"],
        "ICIR": lambda m: m.get("ICIR", 0) >= gates["ICIR"],
        "AnnReturn": lambda m: m.get("AnnReturn", 0) >= gates["AnnReturn"],
        "MaxDrawdown": lambda m: m.get("MaxDrawdown", 0) >= gates["MaxDrawdown"],
        "WinRate": lambda m: m.get("WinRate", 0) >= gates["WinRate"],
        "IC_Positive_Ratio": lambda m: m.get("IC_Positive_Ratio", 0) >= gates["IC_Positive_Ratio"],
    }

    for name, check_fn in gate_checks.items():
        passed = check_fn(metrics)
        actual = metrics.get(name.replace("IC_Positive_Ratio", "IC_Positive_Ratio"), None)
        if actual is None:
            # Map metric names
            if name == "IC":
                actual = metrics.get("Mean_IC")
            elif name == "ICIR":
                actual = metrics.get("ICIR")
            elif name == "AnnReturn":
                actual = metrics.get("AnnReturn")
            elif name == "MaxDrawdown":
                actual = metrics.get("MaxDrawdown")
            elif name == "WinRate":
                actual = metrics.get("WinRate")
            elif name == "IC_Positive_Ratio":
                actual = metrics.get("IC_Positive_Ratio")

        results[name] = {
            "passed": passed,
            "actual": actual,
            "threshold": gates[name],
        }
        if not passed:
            all_pass = False

    return all_pass, results


# ============================================================
# 报告输出
# ============================================================
def print_report(test_metrics, yearly_results, regime_results, gate_results=None, gate_pass=None):
    """打印综合评估报告"""
    print("\n" + "=" * 70)
    print("QuantaAlpha 综合评估报告")
    print("=" * 70)

    # 核心指标
    print("\n【核心指标】（测试集）")
    print(f"  平均IC:          {test_metrics.get('Mean_IC', 'N/A'):.4f}" if isinstance(test_metrics.get('Mean_IC'), float) else f"  平均IC:          N/A")
    print(f"  IC标准差:        {test_metrics.get('IC_Std', 'N/A'):.4f}" if isinstance(test_metrics.get('IC_Std'), float) else f"  IC标准差:        N/A")
    print(f"  ICIR:            {test_metrics.get('ICIR', 'N/A'):.4f}" if isinstance(test_metrics.get('ICIR'), float) else f"  ICIR:            N/A")
    print(f"  IC>0比例:        {test_metrics.get('IC_Positive_Ratio', 'N/A'):.2%}" if isinstance(test_metrics.get('IC_Positive_Ratio'), float) else f"  IC>0比例:        N/A")
    print(f"  年化收益:        {test_metrics.get('AnnReturn', 'N/A'):.2%}" if isinstance(test_metrics.get('AnnReturn'), float) else f"  年化收益:        N/A")
    print(f"  最大回撤:        {test_metrics.get('MaxDrawdown', 'N/A'):.2%}" if isinstance(test_metrics.get('MaxDrawdown'), float) else f"  最大回撤:        N/A")
    print(f"  夏普比率:        {test_metrics.get('Sharpe', 'N/A'):.2f}" if isinstance(test_metrics.get('Sharpe'), float) else f"  夏普比率:        N/A")
    print(f"  胜率:            {test_metrics.get('WinRate', 'N/A'):.2%}" if isinstance(test_metrics.get('WinRate'), float) else f"  胜率:            N/A")
    print(f"  交易日数:        {test_metrics.get('N_TradingDays', 'N/A')}")

    # 分年度
    print("\n【分年度评估】")
    for year, metrics in sorted(yearly_results.items()):
        ic_str = f"{metrics['IC']:.4f}" if metrics['IC'] is not None else "N/A"
        pr_str = f"{metrics['IC_Positive_Ratio']:.2%}" if isinstance(metrics.get('IC_Positive_Ratio'), float) else "N/A"
        print(f"  {year}: IC={ic_str}, 交易日={metrics['N_Days']}, IC>0={pr_str}")

    # 分regime
    print("\n【分Regime评估】")
    for regime in ["Bull", "Bear", "Sideways"]:
        if regime not in regime_results:
            continue
        m = regime_results[regime]
        ic_str = f"{m['IC']:.4f}" if m['IC'] is not None else "N/A"
        ann_str = f"{m['AnnReturn']:.2%}" if isinstance(m.get('AnnReturn'), float) else "N/A"
        wr_str = f"{m['WinRate']:.2%}" if isinstance(m.get('WinRate'), float) else "N/A"
        print(f"  {regime:8s}: IC={ic_str}, 年化={ann_str}, 胜率={wr_str}, 样本={m['N_Samples']}")
        if m.get('Note'):
            print(f"           Note: {m['Note']}")

    # 质量门控
    if gate_results is not None:
        print("\n【质量门控】（v2.0标准）")
        for name, result in sorted(gate_results.items()):
            status = "PASS" if result['passed'] else "FAIL"
            actual_str = f"{result['actual']:.4f}" if isinstance(result['actual'], float) else "N/A"
            threshold_str = f"{result['threshold']:.4f}" if isinstance(result['threshold'], float) else "N/A"
            icon = "✅" if result['passed'] else "❌"
            print(f"  {icon} {name:20s}: {actual_str} (threshold: {threshold_str}) [{status}]")

        if gate_pass:
            print(f"\n  🎉 全部通过！模型符合v2.0质量标准")
        else:
            print(f"\n  ⚠️  未通过全部质量门控，需要优化")


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()

    print("=" * 70)
    print("QuantaAlpha 综合评估")
    print("=" * 70)

    # 默认因子表达式（v1.0三因子）
    if args.factors:
        factor_expressions = [f.strip() for f in args.factors.split(',')]
    else:
        factor_expressions = [
            # 因子1: Hurst_Proxy
            "(0 - (Log($close) - Mean(Log($close), 20)) / (Std(Log($close), 20) + 1e-8)) * Std($close/Ref($close,1)-1, 5) / (Std($close/Ref($close,1)-1, 20) + 1e-8)",
            # 因子2: AR1_Reversion
            "(0 - ($close - Mean($close, 20)) / (Std($close, 20) + 1e-8)) * Sign(0 - Corr($close/Ref($close,1)-1, Ref($close/Ref($close,1)-1, 1), 20))",
            # 因子3: OU_MeanReversion
            "(0 - ($close - Mean($close, 20)) / (Std($close, 20) + 1e-8)) * (1 - Corr($close/Ref($close,1)-1, Ref($close/Ref($close,1)-1, 1), 10))",
        ]

    print(f"\n因子数量: {len(factor_expressions)}")
    for i, expr in enumerate(factor_expressions, 1):
        print(f"  因子{i}: {expr[:80]}...")

    # 1. 加载全量数据
    print("\nStep 1: 加载数据...")
    data_df = load_data(factor_expressions, args.train_start, args.test_end)

    # 2. 切分数据集
    print("\nStep 2: 切分数据集...")
    train_df, valid_df, test_df = split_data(
        data_df, args.train_start, args.train_end,
        args.valid_start, args.valid_end,
        args.test_start, args.test_end
    )

    # 3. 加载模型
    model_path = project_root / args.model_file
    print(f"\nStep 3: 加载模型 ({model_path})...")
    model = load_model(model_path)

    # 4. 特征提取和预测
    print("\nStep 4: 计算预测值...")
    factor_cols = [f'factor_{i+1}' for i in range(len(factor_expressions))]

    X_test = test_df[factor_cols].values
    y_test = test_df['label'].values
    dates_test = test_df.index.get_level_values(1).values

    y_test_pred = model.predict(X_test)
    print(f"  测试集预测完成: {len(y_test_pred)} 样本")

    # 5. 计算每日IC
    print("\nStep 5: 计算每日IC...")
    daily_ic_result = compute_daily_ic(y_test_pred, y_test, dates_test)
    if daily_ic_result:
        print(f"  平均IC: {daily_ic_result['Mean_IC']:.4f}")
        print(f"  ICIR:   {daily_ic_result['ICIR']:.4f}")
        print(f"  IC>0:   {daily_ic_result['IC_Positive_Ratio']:.2%}")

    # 6. 计算组合指标
    print("\nStep 6: 计算Top-K组合指标...")
    portfolio_metrics = compute_portfolio_metrics(y_test_pred, y_test, dates_test, top_k=args.top_k)
    if portfolio_metrics:
        print(f"  年化收益: {portfolio_metrics['AnnReturn']:.2%}")
        print(f"  最大回撤: {portfolio_metrics['MaxDrawdown']:.2%}")
        print(f"  夏普比率: {portfolio_metrics['Sharpe']:.2f}")
        print(f"  胜率:     {portfolio_metrics['WinRate']:.2%}")

    # 合并指标
    test_metrics = {**daily_ic_result, **portfolio_metrics}

    # 7. 分年度评估
    print("\nStep 7: 分年度评估...")
    yearly_results = compute_yearly_breakdown(y_test_pred, y_test, dates_test, top_k=args.top_k)

    # 8. 分regime评估
    print("\nStep 8: 分Regime评估...")
    regime_results = compute_regime_metrics(y_test_pred, y_test, dates_test, top_k=args.top_k)

    # 9. 质量门控
    gate_pass = None
    gate_results = None
    if not args.no_gate:
        gate_pass, gate_results = check_quality_gates(test_metrics)

    # 10. 输出报告
    print_report(test_metrics, yearly_results, regime_results, gate_results, gate_pass)

    # 11. 保存报告
    output_dir = project_root / "evaluation_reports"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"eval_report_{timestamp}.json"

    # 序列化报告（排除IC_Series等pandas对象）
    serializable = {
        "test_metrics": {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in test_metrics.items() if k != 'IC_Series'
        },
        "yearly_results": yearly_results,
        "regime_results": regime_results,
        "quality_gate": {
            "passed": gate_pass,
            "details": {
                k: {
                    "passed": v["passed"],
                    "actual": float(v["actual"]) if isinstance(v["actual"], (np.floating, np.integer)) else v["actual"],
                    "threshold": float(v["threshold"]) if isinstance(v["threshold"], (np.floating, np.integer)) else v["threshold"],
                }
                for k, v in gate_results.items()
            } if gate_results else None,
        },
        "config": {
            "model_file": args.model_file,
            "train_range": [args.train_start, args.train_end],
            "valid_range": [args.valid_start, args.valid_end],
            "test_range": [args.test_start, args.test_end],
            "top_k": args.top_k,
            "n_factors": len(factor_expressions),
        },
    }

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n📄 报告已保存: {report_file}")

    return 0 if (gate_pass or args.no_gate) else 1


if __name__ == "__main__":
    sys.exit(main())
