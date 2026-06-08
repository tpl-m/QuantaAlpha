#!/usr/bin/env python3
"""
将模型转换为pickle protocol=4格式（兼容Python 3.4+）
"""

import pickle
from pathlib import Path

def convert_model():
    """转换模型到兼容协议"""
    model_file = Path(__file__).parent / "exported_models/quantaalpha_model.pkl"
    output_file = Path(__file__).parent / "exported_models/quantaalpha_model_compat.pkl"

    print("加载原模型...")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    print(f"模型类型: {type(model).__name__}")

    print("\n保存为protocol=4格式...")
    with open(output_file, 'wb') as f:
        pickle.dump(model, f, protocol=4)

    print(f"✅ 转换完成")
    print(f"   输出: {output_file}")
    print(f"   大小: {output_file.stat().st_size / 1024:.2f} KB")

    # 验证
    print("\n验证新文件...")
    with open(output_file, 'rb') as f:
        model_test = pickle.load(f)

    import numpy as np
    test_input = np.random.randn(1, 3)
    pred = model_test.predict(test_input)

    print(f"✅ 验证成功，预测值: {pred[0]:.6f}")

    print("\n使用说明:")
    print(f"  上传 {output_file.name} 到聚宽平台（而非原文件）")

if __name__ == '__main__':
    convert_model()
