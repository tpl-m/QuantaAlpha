#!/bin/bash
# 验证生产版本完整性
set -e
VERSION_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "验证 ${VERSION_DIR}..."
echo ""

PASS=0
FAIL=0

check_file() {
    if [ -f "${VERSION_DIR}/$1" ]; then
        echo "  ✅ $1"
        PASS=$((PASS+1))
    else
        echo "  ❌ $1 缺失"
        FAIL=$((FAIL+1))
    fi
}

echo "模型文件:"
check_file "models/quantaalpha_model.txt"
check_file "models/model_metadata.txt"

echo ""
echo "训练脚本:"
check_file "algorithm/train_model_simple.py"
check_file "algorithm/smoke_test.py"

echo ""
echo "聚宽脚本:"
check_file "joinquant_scripts/load_model_joinquant.py"

echo ""
echo "文档:"
check_file "VERSION.txt"
check_file "docs/factor_selection_report.json"
check_file "docs/selected_factors_config.json"

echo ""
echo "========================================"
echo "结果: ${PASS} 通过, ${FAIL} 失败"
if [ ${FAIL} -eq 0 ]; then
    echo "✅ 验证通过！"
    exit 0
else
    echo "❌ 验证失败！"
    exit 1
fi
