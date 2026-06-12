#!/usr/bin/env python3
"""Run comprehensive evaluation with v4.0 factors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Hardcoded v4.0 factors (to avoid shell comma-splitting issues)
V4_FACTORS = [
    "((($open - Ref($close, 1)) / (Mean($high - $low, 10) + 1e-8) * ($high - $low) / (Mean($high - $low, 10) + 1e-8) - Mean(($open - Ref($close, 1)) / (Mean($high - $low, 10) + 1e-8) * ($high - $low) / (Mean($high - $low, 10) + 1e-8), 5)) / (Std(($open - Ref($close, 1)) / (Mean($high - $low, 10) + 1e-8) * ($high - $low) / (Mean($high - $low, 10) + 1e-8), 5) + 1e-8))",
    "Rank(Abs(($close - $low) / ($high - $low + 1e-8) - 0.5), 10)",
    "Corr(($open - Ref($close, 1)) / (Ref($close, 1) + 1e-8), ($close - $low) / ($high - $low + 1e-8), 20)",
    "Rank((Mean(Delta($volume, 1), 10) - Mean(Delta($volume, 1), 20)) / (Std(Delta($volume, 1), 20) + 1e-8), 60)",
    "Count(Abs(($close - Ref($close, 1)) / (Ref($close, 1) + 1e-8) - Mean(($close - Ref($close, 1)) / (Ref($close, 1) + 1e-8), 20)) > 1.5 * Std(($close - Ref($close, 1)) / (Ref($close, 1) + 1e-8), 20), 5) / 5",
]

if __name__ == "__main__":
    sys.argv = [
        "evaluate_model_comprehensive.py",
        "--model-file", "exported_models/quantaalpha_model.txt",
        "--factors", "|".join(V4_FACTORS),  # Use | as separator to avoid comma issues
    ]

    # Monkey-patch the split separator
    import evaluate_model_comprehensive as eval_mod
    orig_parse = eval_mod.parse_args

    def patched_parse():
        import argparse
        parser = argparse.ArgumentParser(description="QuantaAlpha模型综合评估")
        parser.add_argument("--model-file", type=str, default="exported_models/quantaalpha_model.txt")
        parser.add_argument("--factors", type=str, default=None)
        parser.add_argument("--train-start", type=str, default="2016-01-01")
        parser.add_argument("--train-end", type=str, default="2023-12-31")
        parser.add_argument("--valid-start", type=str, default="2024-01-01")
        parser.add_argument("--valid-end", type=str, default="2024-12-31")
        parser.add_argument("--test-start", type=str, default="2025-01-01")
        parser.add_argument("--test-end", type=str, default="2025-12-26")
        parser.add_argument("--top-k", type=int, default=50)
        parser.add_argument("--no-gate", action="store_true")
        args = parser.parse_args()
        return args

    eval_mod.parse_args = patched_parse

    # Also patch the factor parsing in main()
    orig_main = eval_mod.main

    def patched_main():
        args = patched_parse()

        # Use hardcoded factors instead of args.factors
        factor_expressions = V4_FACTORS

        print("=" * 70)
        print("QuantaAlpha 综合评估")
        print("=" * 70)

        print(f"\n因子数量: {len(factor_expressions)}")
        for i, expr in enumerate(factor_expressions, 1):
            print(f"  因子{i}: {expr[:80]}...")

        # 1. 加载全量数据
        print("\nStep 1: 加载数据...")
        data_df = eval_mod.load_data(factor_expressions, args.train_start, args.test_end)

        # 2. 切分数据集
        print("\nStep 2: 切分数据集...")
        train_df, valid_df, test_df = eval_mod.split_data(
            data_df, args.train_start, args.train_end,
            args.valid_start, args.valid_end,
            args.test_start, args.test_end
        )

        # 3. 加载模型
        model_path = Path(__file__).parent / args.model_file
        print(f"\nStep 3: 加载模型 ({model_path})...")
        model = eval_mod.load_model(model_path)

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
        daily_ic_result = eval_mod.compute_daily_ic(y_test_pred, y_test, dates_test)
        if daily_ic_result:
            print(f"  平均IC: {daily_ic_result['Mean_IC']:.4f}")
            print(f"  ICIR:   {daily_ic_result['ICIR']:.4f}")
            print(f"  IC>0:   {daily_ic_result['IC_Positive_Ratio']:.2%}")

        # 6. 计算组合指标
        print("\nStep 6: 计算Top-K组合指标...")
        portfolio_metrics = eval_mod.compute_portfolio_metrics(y_test_pred, y_test, dates_test, top_k=args.top_k)
        if portfolio_metrics:
            print(f"  年化收益: {portfolio_metrics['AnnReturn']:.2%}")
            print(f"  最大回撤: {portfolio_metrics['MaxDrawdown']:.2%}")
            print(f"  夏普比率: {portfolio_metrics['Sharpe']:.2f}")
            print(f"  胜率:     {portfolio_metrics['WinRate']:.2%}")

        test_metrics = {**daily_ic_result, **portfolio_metrics}

        # 7. 分年度评估
        print("\nStep 7: 分年度评估...")
        yearly_results = eval_mod.compute_yearly_breakdown(y_test_pred, y_test, dates_test, top_k=args.top_k)

        # 8. 分regime评估
        print("\nStep 8: 分Regime评估...")
        regime_results = eval_mod.compute_regime_metrics(y_test_pred, y_test, dates_test, top_k=args.top_k)

        # 9. 质量门控
        gate_pass = None
        gate_results = None
        if not args.no_gate:
            gate_pass, gate_results = eval_mod.check_quality_gates(test_metrics)

        # 10. 输出报告
        eval_mod.print_report(test_metrics, yearly_results, regime_results, gate_results, gate_pass)

        # 11. 保存报告
        import json
        from datetime import datetime
        output_dir = Path(__file__).parent / "evaluation_reports"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"eval_report_{timestamp}.json"

        serializable = {
            "test_metrics": {
                k: (float(v) if hasattr(v, 'item') else v)
                for k, v in test_metrics.items() if k != 'IC_Series'
            },
            "yearly_results": yearly_results,
            "regime_results": regime_results,
            "quality_gate": {
                "passed": gate_pass,
                "details": {
                    k: {
                        "passed": v["passed"],
                        "actual": float(v["actual"]) if hasattr(v["actual"], 'item') else v["actual"],
                        "threshold": float(v["threshold"]) if hasattr(v["threshold"], 'item') else v["threshold"],
                    }
                    for k, v in gate_results.items()
                } if gate_results else None,
            },
            "config": {
                "model_file": args.model_file,
                "n_factors": len(factor_expressions),
            },
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n📄 报告已保存: {report_file}")

        return 0 if (gate_pass or args.no_gate) else 1

    sys.exit(patched_main())
