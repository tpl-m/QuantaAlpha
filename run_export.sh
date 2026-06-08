#!/bin/bash
# 在QuantaAlpha虚拟环境中运行模型导出脚本

cd "$(dirname "$0")"

# 激活虚拟环境
source venv/bin/activate

# 运行导出脚本
python3 train_and_export_model.py "$@"
