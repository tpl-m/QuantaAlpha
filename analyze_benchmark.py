#!/usr/bin/env python3
"""
分析回测结果中的基准收益率
从CSV文件中读取数据并计算基准的年化收益
"""
import sys
import json
from pathlib import Path

# 添加qlib环境
sys.path.insert(0, str(Path.home() / 'miniforge3/envs/qlib/lib/python3.10/site-packages'))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("请在qlib环境中运行此脚本:")
    print("  conda activate qlib")
    print("  python3 analyze_benchmark.py")
    sys.exit(1)


def analyze_csv(csv_path):
    """分析CSV文件"""
    print(f"\n{'='*70}")
    print(f"分析文件: {csv_path.name}")
    print(f"{'='*70}")

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"\n📊 文件列: {df.columns.tolist()}")

        # 基本信息
        start_date = df.index[0]
        end_date = df.index[-1]
        trading_days = len(df)
        years = trading_days / 252

        print(f"\n📅 测试期信息:")
        print(f"  起始日期: {start_date.date()}")
        print(f"  结束日期: {end_date.date()}")
        print(f"  交易日数: {trading_days}天")
        print(f"  年数: {years:.2f}年")

        # 检查是否有新增的列
        if 'benchmark_return' in df.columns:
            print(f"\n✅ 文件包含基准数据!")

            # 计算基准年化收益
            bench_cum = df['cumulative_benchmark'].iloc[-1]
            bench_ann = (1 + bench_cum) ** (1/years) - 1

            print(f"\n📈 沪深300基准收益:")
            print(f"  累计收益: {bench_cum*100:.2f}%")
            print(f"  年化收益: {bench_ann*100:.2f}%")

            # 计算组合年化收益
            if 'cumulative_portfolio' in df.columns:
                port_cum = df['cumulative_portfolio'].iloc[-1]
                port_ann = (1 + port_cum) ** (1/years) - 1

                print(f"\n📊 因子组合收益:")
                print(f"  累计收益: {port_cum*100:.2f}%")
                print(f"  年化收益: {port_ann*100:.2f}%")

                # 超额收益
                excess_cum = df['cumulative_excess_return'].iloc[-1]
                excess_ann = (1 + excess_cum) ** (1/years) - 1

                print(f"\n💎 超额收益:")
                print(f"  累计超额: {excess_cum*100:.2f}%")
                print(f"  年化超额: {excess_ann*100:.2f}%")

                print(f"\n🎯 收益对比:")
                print(f"  基准年化: {bench_ann*100:>7.2f}%")
                print(f"  组合年化: {port_ann*100:>7.2f}%")
                print(f"  超额年化: {excess_ann*100:>7.2f}%")
                print(f"  验证: {port_ann*100:.2f}% - {bench_ann*100:.2f}% ≈ {excess_ann*100:.2f}%")

        else:
            print(f"\n⚠️ 文件只包含超额收益，缺少基准原始数据")
            print(f"需要重新运行回测以生成完整数据")

            # 只能显示超额收益
            excess_cum = df['cumulative_excess_return'].iloc[-1]
            excess_ann = (1 + excess_cum) ** (1/years) - 1

            print(f"\n📈 超额收益（已有数据）:")
            print(f"  累计超额: {excess_cum*100:.2f}%")
            print(f"  年化超额: {excess_ann*100:.2f}%")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


def analyze_json(json_path):
    """分析JSON结果文件"""
    print(f"\n{'='*70}")
    print(f"分析JSON: {json_path.name}")
    print(f"{'='*70}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metrics = data.get('metrics', {})
        config = data.get('config', {})

        print(f"\n📋 配置信息:")
        print(f"  测试范围: {config.get('test_range')}")
        print(f"  回测范围: {config.get('backtest_range')}")
        print(f"  市场: {config.get('market')}")
        print(f"  基准: {config.get('benchmark')}")

        print(f"\n📊 指标:")

        # 检查是否有新增的基准指标
        if 'benchmark_annualized_return' in metrics:
            print(f"\n✅ JSON包含基准数据!")
            print(f"  基准年化收益: {metrics['benchmark_annualized_return']*100:.2f}%")
            print(f"  组合年化收益: {metrics['portfolio_annualized_return']*100:.2f}%")
            print(f"  超额年化收益: {metrics['annualized_return']*100:.2f}%")
        else:
            print(f"\n⚠️ JSON只包含超额收益")
            print(f"  超额年化收益: {metrics.get('annualized_return', 'N/A')}")

        print(f"\n  IC: {metrics.get('IC', 'N/A')}")
        print(f"  ICIR: {metrics.get('ICIR', 'N/A')}")
        print(f"  信息比率: {metrics.get('information_ratio', 'N/A')}")
        print(f"  最大回撤: {metrics.get('max_drawdown', 'N/A')}")

    except Exception as e:
        print(f"❌ JSON分析失败: {e}")


if __name__ == "__main__":
    # 查找最近的回测结果
    result_dir = Path("data/results/backtest_v2_results")

    if not result_dir.exists():
        print(f"❌ 结果目录不存在: {result_dir}")
        sys.exit(1)

    # 分析所有CSV文件
    csv_files = sorted(result_dir.glob("*_cumulative_excess.csv"))

    if not csv_files:
        print(f"❌ 未找到回测结果CSV文件")
        sys.exit(1)

    print(f"找到 {len(csv_files)} 个CSV文件")

    for csv_file in csv_files:
        analyze_csv(csv_file)

        # 查找对应的JSON
        json_file = csv_file.with_suffix('.csv').parent / csv_file.name.replace('_cumulative_excess.csv', '_backtest_metrics.json')
        if json_file.exists():
            analyze_json(json_file)

    print(f"\n{'='*70}")
    print("分析完成")
    print(f"{'='*70}\n")
