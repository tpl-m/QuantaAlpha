#!/usr/bin/env python3
"""
验证导出的模型文件是否可以正常加载

在上传聚宽前运行此脚本，确保模型文件正确
"""

import pickle
import numpy as np
from pathlib import Path

def verify_model_file():
    """验证模型文件"""
    print("=" * 70)
    print("验证QuantaAlpha模型文件")
    print("=" * 70)

    model_file = Path(__file__).parent / "exported_models/quantaalpha_model.pkl"

    # 检查文件存在
    if not model_file.exists():
        print(f"\n❌ 模型文件不存在: {model_file}")
        print("请先运行 train_model_simple.py 训练模型")
        return False

    print(f"\n✅ 模型文件存在")
    print(f"   路径: {model_file}")
    print(f"   大小: {model_file.stat().st_size / 1024:.2f} KB")

    # 尝试加载模型
    print("\n正在加载模型...")
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print("✅ 使用pickle加载成功")
    except Exception as e:
        print(f"❌ pickle加载失败: {e}")
        return False

    # 检查模型类型
    print(f"\n模型信息:")
    print(f"  类型: {type(model).__name__}")

    # 检查LightGBM特有属性
    if hasattr(model, 'num_trees'):
        print(f"  树数量: {model.num_trees()}")
    else:
        print("  ⚠️  警告: 不是LightGBM模型")

    if hasattr(model, 'num_feature'):
        print(f"  特征数: {model.num_feature()}")
    else:
        print("  ⚠️  警告: 无法获取特征数量")

    # 测试预测
    print("\n测试预测功能...")
    try:
        # 创建测试输入（3个特征）
        test_input = np.random.randn(1, 3)
        prediction = model.predict(test_input)

        print(f"✅ 预测成功")
        print(f"   输入shape: {test_input.shape}")
        print(f"   输出: {prediction[0]:.6f}")

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return False

    # 检查pickle协议版本
    print("\n检查pickle协议版本...")
    with open(model_file, 'rb') as f:
        first_byte = f.read(1)
        protocol = ord(first_byte)

    print(f"   Pickle协议版本: {protocol}")

    if protocol <= 4:
        print(f"   ✅ 协议版本{protocol}兼容Python 3.4+")
    else:
        print(f"   ⚠️  协议版本{protocol}可能不兼容旧版本Python")
        print(f"   建议使用protocol=4重新保存")

    # 最终检查
    print("\n" + "=" * 70)
    print("验证完成！")
    print("=" * 70)

    print("\n✅ 模型文件可以正常使用")
    print("\n下一步：")
    print("  1. 登录聚宽平台（企业版）")
    print("  2. 上传 exported_models/quantaalpha_model.pkl")
    print("  3. 运行策略脚本 factor_combined_meanreversion.py")

    return True


if __name__ == '__main__':
    verify_model_file()
