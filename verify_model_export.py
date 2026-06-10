#!/usr/bin/env python3
"""
验证导出的模型文件是否可以正常加载

在上传聚宽前运行此脚本，确保模型文件正确
"""

import numpy as np
from pathlib import Path

def verify_model_file():
    """验证模型文件"""
    print("=" * 70)
    print("验证QuantaAlpha模型文件")
    print("=" * 70)

    exported_dir = Path(__file__).parent / "exported_models"
    model_file = exported_dir / "quantaalpha_model.pkl"
    txt_file = exported_dir / "quantaalpha_model.txt"

    # 检查主要生产文件（.txt）
    print("\n--- 主要生产格式 (.txt) ---")
    if not txt_file.exists():
        print(f"❌ .txt模型文件不存在: {txt_file}")
        print("请先运行 train_model_simple.py 训练并导出模型")
        return False

    print(f"✅ .txt模型文件存在")
    print(f"   路径: {txt_file}")
    print(f"   大小: {txt_file.stat().st_size / 1024:.2f} KB")

    # 验证.txt文件可加载（聚宽部署门控）
    print("\n正在验证.txt模型加载（聚宽部署门控）...")
    try:
        import lightgbm as lgb
        import tempfile
        import os
        import contextlib
        with open(txt_file, 'rb') as f:
            model_bytes = f.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='wb')
        tmp_path = tmp.name
        try:
            with tmp:  # guarantees close even if write/flush raises
                tmp.write(model_bytes)
                tmp.flush()
            lgb_model = lgb.Booster(model_file=tmp_path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
        print("✅ .txt格式加载成功（lgb.Booster）")
        print(f"   树数量: {lgb_model.num_trees()}")
        print(f"   特征数: {lgb_model.num_feature()}")
    except Exception as e:
        print(f"❌ .txt格式加载失败: {e}")
        print("⭐ .txt格式是聚宽部署的主要门控，此错误需修复才能部署")
        return False

    # 检查备用格式（.pkl）— 本地验证用，非必需
    print("\n--- 备用格式 (.pkl，本地验证用）---")
    if not model_file.exists():
        print(f"⚠️  .pkl文件不存在: {model_file}")
        print("   （备用格式，不影响聚宽部署）")
    else:
        print(f"✅ .pkl文件存在")
        print(f"   路径: {model_file}")
        print(f"   大小: {model_file.stat().st_size / 1024:.2f} KB")

        # 检查pickle协议版本
        print("\n检查pickle协议版本...")
        with open(model_file, 'rb') as f:
            header = f.read(2)

        # Binary pickles (protocol >= 2) start with opcode 0x80 followed by the protocol byte.
        # Reading only the first byte returns 0x80 (=128), not the actual protocol number.
        if len(header) >= 2 and header[0] == 0x80:
            protocol = header[1]
        else:
            protocol = 0  # protocol 0 or 1 — text-based, no 0x80 prefix

        print(f"   Pickle协议版本: {protocol}")

        if protocol <= 4:
            print(f"   ✅ 协议版本{protocol}兼容Python 3.4+")
        else:
            print(f"   ⚠️  协议版本{protocol}可能不兼容旧版本Python")
            print(f"   建议使用protocol=4重新保存")

    # 测试预测功能（使用.txt加载的模型）
    print("\n测试预测功能（基于.txt模型）...")
    try:
        test_input = np.random.randn(1, 3)
        prediction = lgb_model.predict(test_input)

        print(f"✅ 预测成功")
        print(f"   输入shape: {test_input.shape}")
        print(f"   输出: {prediction[0]:.6f}")

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return False

    # 最终检查
    print("\n" + "=" * 70)
    print("验证完成！")
    print("=" * 70)

    print("\n✅ 模型文件可以正常使用")
    print("\n下一步：")
    print("  1. 确认 exported_models/quantaalpha_model.txt 存在（主要生产格式）✅ 已验证")
    print("  2. 运行 test_txt_loading.py 验证 .txt 格式可加载（⭐ 主要门控）✅ 本脚本已验证")
    print("  3. 将 exported_models/quantaalpha_model.txt 上传至聚宽平台")
    print("  4. 运行策略脚本 jukuan_scrips/factor_combined_meanreversion.py")

    return True


if __name__ == '__main__':
    verify_model_file()
