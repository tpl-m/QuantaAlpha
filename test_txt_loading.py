#!/usr/bin/env python3
"""
测试LightGBM .txt格式加载（模拟聚宽环境）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_txt_loading():
    """模拟聚宽平台的加载方式"""
    print("=" * 70)
    print("测试LightGBM .txt格式加载（模拟聚宽环境）")
    print("=" * 70)

    model_file = Path(__file__).parent / "exported_models/quantaalpha_model.txt"

    if not model_file.exists():
        print(f"\n❌ 模型文件不存在: {model_file}")
        return False

    print(f"\n✅ 模型文件存在: {model_file.name}")
    print(f"   大小: {model_file.stat().st_size / 1024:.2f} KB")

    # 模拟聚宽的read_file()读取
    print("\n步骤1: 读取模型文件（模拟read_file）...")
    with open(model_file, 'rb') as f:
        model_bytes = f.read()

    print(f"✅ 读取成功，大小: {len(model_bytes)/1024:.2f} KB")

    # 模拟聚宽的加载方式：写入临时文件 → 用lgb.Booster加载
    print("\n步骤2: 写入临时文件...")
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='wb') as f:
        f.write(model_bytes)
        temp_path = f.name

    print(f"✅ 临时文件: {temp_path}")

    print("\n步骤3: 使用lgb.Booster加载（聚宽方式）...")
    try:
        import lightgbm as lgb

        model = lgb.Booster(model_file=temp_path)
        print("✅ LightGBM原生格式加载成功")

        # 验证模型
        print(f"\n模型信息:")
        print(f"  树数量: {model.num_trees()}")
        print(f"  特征数: {model.num_feature()}")

        # 测试预测
        import numpy as np
        test_input = np.random.randn(1, 3)
        pred = model.predict(test_input)

        print(f"\n✅ 预测测试:")
        print(f"  输入shape: {test_input.shape}")
        print(f"  输出: {pred[0]:.6f}")

        print("\n" + "=" * 70)
        print("✅ .txt格式完全兼容聚宽加载方式！")
        print("=" * 70)

        print("\n下一步:")
        print("  1. 上传 exported_models/quantaalpha_model.txt 到聚宽")
        print("  2. 使用 factor_combined_meanreversion.py 脚本")
        print("  3. MODEL_FILE已默认配置为 'quantaalpha_model.txt'")

        return True

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    test_txt_loading()
