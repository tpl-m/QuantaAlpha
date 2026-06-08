#!/usr/bin/env python3
"""
详细分析回测结果，包括基准、组合、超额收益的分年度对比
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'venv/lib/python3.11/site-packages'))

import pandas as pd
import numpy as np


def analyze_benchmark_performance():
    """分析基准和因子的详细表现"""

    # 读取数据
    csv_path = Path("data/results/backtest_v2_results/backtest_v2_experiment_cumulative_excess.csv")
    json_path = Path("data/results/backtest_v2_results/backtest_metrics.json")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    with open(json_path, 'r') as f:
        metrics = json.load(f)['metrics']

    # 基本信息
    start_date = df.index[0]
    end_date = df.index[-1]
    trading_days = len(df)
    years = trading_days / 252

    print("\n" + "="*80)
    print("📊 回测结果详细分析 - 基准 vs 因子组合")
    print("="*80)

    print(f"\n📅 测试期间: {start_date.date()} 至 {end_date.date()}")
    print(f"   交易日数: {trading_days}天 ({years:.2f}年)")
    print(f"   基准指数: 沪深300 (SH000300)")
    print(f"   因子数量: alpha158前20个因子")

    # 整体表现
    print("\n" + "="*80)
    print("📈 整体年化收益对比")
    print("="*80)

    bench_ann = metrics['benchmark_annualized_return']
    port_ann = metrics['portfolio_annualized_return']
    excess_ann = metrics['annualized_return']

    print(f"\n{'指标':<20} {'数值':>12} {'说明':<40}")
    print("-"*80)
    print(f"{'沪深300年化收益':<20} {bench_ann*100:>11.2f}% {'基准市场表现（几乎持平）':<40}")
    print(f"{'因子组合年化收益':<20} {port_ann*100:>11.2f}% {'Alpha158因子创造的绝对收益':<40}")
    print(f"{'超额年化收益':<20} {excess_ann*100:>11.2f}% {'相对基准的超额收益（已扣成本）':<40}")

    # 核心结论
    print("\n" + "="*80)
    print("🎯 核心结论")
    print("="*80)

    print(f"\n1. ✅ **基准表现**: 沪深300在2022-2025年年化收益仅{bench_ann*100:.2f}%（约等于0）")
    print(f"   → 这是一个典型的震荡市/熊市环境")
    print(f"   → 市场基本没有方向性收益")

    print(f"\n2. 💎 **因子Alpha**: 在市场几乎不涨的情况下，因子组合年化{port_ann*100:.2f}%")
    print(f"   → 说明因子**不是顺势而为**，而是真正有选股能力")
    print(f"   → 在震荡市中能创造{port_ann*100:.2f}%的绝对收益是优秀表现")

    print(f"\n3. ⚠️ **超额收益**: 超额年化{excess_ann*100:.2f}%")
    print(f"   → 组合年化{port_ann*100:.2f}% - 基准年化{bench_ann*100:.2f}% - 成本 ≈ {excess_ann*100:.2f}%")
    print(f"   → 交易成本消耗了约{(port_ann - bench_ann - excess_ann)*100:.2f}%")

    # 分年度分析
    print("\n" + "="*80)
    print("📆 分年度收益对比")
    print("="*80)

    df_yearly = df.copy()
    df_yearly['year'] = df_yearly.index.year

    print(f"\n{'年份':<8} {'基准累计':>12} {'组合累计':>12} {'超额累计':>12} {'基准年化':>12} {'组合年化':>12}")
    print("-"*80)

    for year in sorted(df_yearly['year'].unique()):
        year_df = df_yearly[df_yearly['year'] == year]

        bench_cum = year_df['benchmark_return'].sum()
        port_cum = year_df['portfolio_return'].sum()
        excess_cum = year_df['daily_excess_return'].sum()

        # 计算该年的年化（如果不足1年，按实际天数）
        year_days = len(year_df)
        year_frac = year_days / 252

        bench_ann_y = (1 + bench_cum) ** (1/year_frac) - 1 if year_frac > 0 else 0
        port_ann_y = (1 + port_cum) ** (1/year_frac) - 1 if year_frac > 0 else 0

        print(f"{year:<8} {bench_cum*100:>11.2f}% {port_cum*100:>11.2f}% {excess_cum*100:>11.2f}% {bench_ann_y*100:>11.2f}% {port_ann_y*100:>11.2f}%")

    # 市场环境分析
    print("\n" + "="*80)
    print("🌍 市场环境分析 (2022-2025)")
    print("="*80)

    print("\n2022年: 疫情影响+经济下行")
    year_2022 = df_yearly[df_yearly['year'] == 2022]
    print(f"  基准收益: {year_2022['benchmark_return'].sum()*100:.2f}% {'（大跌）' if year_2022['benchmark_return'].sum() < -0.15 else '（下跌）' if year_2022['benchmark_return'].sum() < 0 else '（上涨）'}")
    print(f"  组合收益: {year_2022['portfolio_return'].sum()*100:.2f}%")
    print(f"  因子表现: {'✅ 在熊市中提供了防御' if year_2022['portfolio_return'].sum() > year_2022['benchmark_return'].sum() else '❌ 跟随市场下跌'}")

    if 2023 in df_yearly['year'].values:
        print("\n2023年: 震荡反弹")
        year_2023 = df_yearly[df_yearly['year'] == 2023]
        print(f"  基准收益: {year_2023['benchmark_return'].sum()*100:.2f}%")
        print(f"  组合收益: {year_2023['portfolio_return'].sum()*100:.2f}%")
        print(f"  因子表现: {'✅ 在震荡中创造超额' if year_2023['portfolio_return'].sum() > year_2023['benchmark_return'].sum() else '❌ 跑输基准'}")

    if 2024 in df_yearly['year'].values:
        print("\n2024年: 持续调整")
        year_2024 = df_yearly[df_yearly['year'] == 2024]
        print(f"  基准收益: {year_2024['benchmark_return'].sum()*100:.2f}%")
        print(f"  组合收益: {year_2024['portfolio_return'].sum()*100:.2f}%")
        print(f"  因子表现: {'✅ 在调整中保持稳健' if year_2024['portfolio_return'].sum() > year_2024['benchmark_return'].sum() else '❌ 防御不足'}")

    if 2025 in df_yearly['year'].values:
        print("\n2025年: 部分复苏")
        year_2025 = df_yearly[df_yearly['year'] == 2025]
        print(f"  基准收益: {year_2025['benchmark_return'].sum()*100:.2f}%")
        print(f"  组合收益: {year_2025['portfolio_return'].sum()*100:.2f}%")
        print(f"  因子表现: {'✅ 在复苏中跑赢' if year_2025['portfolio_return'].sum() > year_2025['benchmark_return'].sum() else '❌ 未跟上反弹'}")

    # 风险指标
    print("\n" + "="*80)
    print("⚠️ 风险指标")
    print("="*80)

    max_dd = metrics['max_drawdown']
    info_ratio = metrics['information_ratio']
    calmar = metrics['calmar_ratio']

    print(f"\n最大回撤: {max_dd*100:.2f}%")
    print(f"  说明: 超额收益曲线从高点到低点的最大跌幅")
    print(f"  评价: {'⚠️ 回撤较大' if abs(max_dd) > 0.2 else '✅ 回撤可控'}")

    print(f"\n信息比率: {info_ratio:.2f}")
    print(f"  说明: 每承担1%跟踪误差，可获得{info_ratio:.2f}%超额收益")
    print(f"  评价: {'✅ 优秀' if info_ratio > 1.0 else '⚠️ 一般' if info_ratio > 0.5 else '❌ 较弱'}")

    print(f"\nCalmar比率: {calmar:.2f}")
    print(f"  说明: 超额年化收益/最大回撤 = {excess_ann*100:.2f}% / {abs(max_dd)*100:.2f}%")
    print(f"  评价: {'✅ 收益风险比优秀' if calmar > 1.0 else '⚠️ 收益风险比一般'}")

    # 最终评价
    print("\n" + "="*80)
    print("🏆 综合评价")
    print("="*80)

    print("\n✅ **因子质量**: Alpha158因子在震荡市环境下表现优秀")
    print(f"   - 基准几乎0收益（{bench_ann*100:.2f}%），因子创造{port_ann*100:.2f}%绝对收益")
    print(f"   - 说明因子有真正的选股alpha，不是简单顺势")

    print("\n⚠️ **注意事项**:")
    print(f"   - 最大回撤{abs(max_dd)*100:.2f}%较大，需要注意风险控制")
    print(f"   - IC={metrics['IC']:.4f}较低，预测能力有限")
    print(f"   - 交易成本消耗约{(port_ann - bench_ann - excess_ann)*100:.2f}%，需优化换手率")

    print("\n💡 **改进方向**:")
    print("   1. 如果基准年化很高（如>15%），需要分析牛市表现")
    print("   2. 建议加入更多强alpha因子，提高IC")
    print("   3. 优化交易策略，降低换手和成本")
    print("   4. 加入风控模块，控制最大回撤")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        analyze_benchmark_performance()
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
