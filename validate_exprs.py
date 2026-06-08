#!/usr/bin/env python3
"""验证三因子Qlib表达式"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Qlib中负号用 0 - expr 实现
FACTOR_EXPRS = [
    # 因子1: Hurst Proxy  = -(ZScore_LogClose) * (Std5 / Std20)
    "(0 - (Log($close) - Mean(Log($close), 20)) / (Std(Log($close), 20) + 1e-8)) * Std($close/Ref($close,1)-1, 5) / (Std($close/Ref($close,1)-1, 20) + 1e-8)",

    # 因子2: AR1 Reversion = -(ZScore_Close) * Sign(-Corr20)
    "(0 - ($close - Mean($close, 20)) / (Std($close, 20) + 1e-8)) * Sign(0 - Corr($close/Ref($close,1)-1, Ref($close/Ref($close,1)-1, 1), 20))",

    # 因子3: OU MeanReversion = -(ZScore_Close) * (1 - Corr10)
    "(0 - ($close - Mean($close, 20)) / (Std($close, 20) + 1e-8)) * (1 - Corr($close/Ref($close,1)-1, Ref($close/Ref($close,1)-1, 1), 10))",
]

FACTOR_NAMES = ['factor1_hurst', 'factor2_ar1', 'factor3_ou']

if __name__ == '__main__':
    import qlib
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')
    from qlib.data import D

    print("验证三因子表达式...")
    test_stocks = ['SZ000001', 'SZ000002', 'SH600000']
    df = D.features(test_stocks, FACTOR_EXPRS, start_time='2022-01-01', end_time='2022-03-31')
    df.columns = FACTOR_NAMES
    print(f"样本数: {len(df)}")
    print(f"NaN: {df.isnull().sum().to_dict()}")
    print(f"\n统计摘要:")
    print(df.describe().to_string())
    print("\n✅ 表达式验证通过！")
