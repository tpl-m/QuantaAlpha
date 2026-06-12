#!/bin/bash
# QuantaAlpha v2.0 生产版本验证脚本
# 检查文件完整性、模型格式、加载正确性

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=0
FAIL=0
WARN=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "  ${GREEN}✅ PASS${NC} $1"
    PASS=$((PASS + 1))
}

check_fail() {
    echo -e "  ${RED}❌ FAIL${NC} $1"
    FAIL=$((FAIL + 1))
}

check_warn() {
    echo -e "  ${YELLOW}⚠️  WARN${NC} $1"
    WARN=$((WARN + 1))
}

echo "======================================================================"
echo "QuantaAlpha v2.0 生产版本验证"
echo "======================================================================"
echo "  目录: $SCRIPT_DIR"
echo "  日期: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================================
# 1. 目录结构检查
# ============================================================
echo "--- 1. 目录结构检查 ---"

for dir in models algorithm joinquant_scripts docs; do
    if [ -d "$SCRIPT_DIR/$dir" ]; then
        check_pass "目录存在: $dir/"
    else
        check_fail "目录缺失: $dir/"
    fi
done

# ============================================================
# 2. 模型文件检查
# ============================================================
echo ""
echo "--- 2. 模型文件检查 ---"

# Check .txt model (primary format for JoinQuant)
if [ -f "$SCRIPT_DIR/models/quantaalpha_model.txt" ]; then
    TXT_SIZE=$(stat -f%z "$SCRIPT_DIR/models/quantaalpha_model.txt" 2>/dev/null || stat -c%s "$SCRIPT_DIR/models/quantaalpha_model.txt" 2>/dev/null || echo "0")
    if [ "$TXT_SIZE" -gt 1000 ]; then
        check_pass "quantaalpha_model.txt 存在 (${TXT_SIZE} bytes)"
    else
        check_fail "quantaalpha_model.txt 文件过小 (${TXT_SIZE} bytes)"
    fi
else
    check_fail "quantaalpha_model.txt 不存在"
fi

# Check model metadata
if [ -f "$SCRIPT_DIR/models/model_metadata.txt" ]; then
    check_pass "model_metadata.txt 存在"
else
    check_warn "model_metadata.txt 不存在（可选）"
fi

# ============================================================
# 3. 算法文件检查
# ============================================================
echo ""
echo "--- 3. 算法文件检查 ---"

for file in train_model_simple.py train_model_optuna.py compute_feature_cache.py smoke_test.py; do
    if [ -f "$SCRIPT_DIR/algorithm/$file" ]; then
        check_pass "算法文件: $file"
    else
        check_fail "算法文件缺失: $file"
    fi
done

# ============================================================
# 4. 聚宽脚本检查
# ============================================================
echo ""
echo "--- 4. 聚宽脚本检查 ---"

if [ -f "$SCRIPT_DIR/joinquant_scripts/load_model_joinquant.py" ]; then
    # Check for required elements
    if grep -q "read_file" "$SCRIPT_DIR/joinquant_scripts/load_model_joinquant.py" 2>/dev/null; then
        check_pass "load_model_joinquant.py 存在（使用read_file API）"
    else
        check_warn "load_model_joinquant.py 存在（未使用read_file API）"
    fi

    if grep -q "after_code_changed" "$SCRIPT_DIR/joinquant_scripts/load_model_joinquant.py" 2>/dev/null; then
        check_pass "包含 after_code_changed 回调"
    else
        check_warn "缺少 after_code_changed 回调（聚宽热更新需要）"
    fi
elif [ -f "$SCRIPT_DIR/joinquant_scripts/README_v5.txt" ]; then
    # v5.0 package: manual loader is expected
    check_pass "v5.0包：README_v5.txt存在（需手动实现因子）"
else
    check_fail "load_model_joinquant.py 不存在"
fi

# ============================================================
# 5. 文档文件检查
# ============================================================
echo ""
echo "--- 5. 文档文件检查 ---"

if [ -f "$SCRIPT_DIR/docs/factor_candidate_pool.json" ]; then
    check_pass "factor_candidate_pool.json 存在"
else
    check_warn "factor_candidate_pool.json 不存在（Phase 1产物）"
fi

if [ -f "$SCRIPT_DIR/docs/best_configuration.json" ]; then
    check_pass "best_configuration.json 存在"
else
    check_warn "best_configuration.json 不存在（Phase 2产物）"
fi

# Check for evaluation report + quality gate (use newest report)
EVAL_REPORT=$(ls -t "$SCRIPT_DIR/docs/eval_report_"*.json 2>/dev/null | head -1)
if [ -n "$EVAL_REPORT" ]; then
    check_pass "评估报告存在: $(basename $EVAL_REPORT)"

    # Check quality gate pass/fail
    if command -v python3 &>/dev/null; then
        GATE_RESULT=$(python3 -c "
import json
try:
    report = json.loads(open('$EVAL_REPORT').read())
    gate = report.get('quality_gate', {})
    passed = gate.get('passed', None)
    if passed is True:
        print('PASS')
    elif passed is False:
        print('FAIL')
    else:
        print('UNKNOWN')
except:
    print('PARSE_ERROR')
" 2>&1)

        if [ "$GATE_RESULT" = "PASS" ]; then
            check_pass "质量门控: 全部通过"
        elif [ "$GATE_RESULT" = "FAIL" ]; then
            check_fail "质量门控: 未通过（模型不满足质量要求）"
            echo "  详情: $(python3 -c "
import json
report = json.loads(open('$EVAL_REPORT').read())
gate = report.get('quality_gate', {})
results = gate.get('results', [])
failed = [r for r in results if not r.get('passed', False)]
for r in failed:
    print(f\"  - {r.get('metric', 'unknown')}: {r.get('actual', 'N/A')} (threshold: {r.get('threshold', 'N/A')})\")
" 2>&1)"
        elif [ "$GATE_RESULT" = "PARSE_ERROR" ]; then
            check_warn "质量门控: 无法解析评估报告"
        else
            check_warn "质量门控: 评估报告格式不完整"
        fi
    fi
else
    check_warn "评估报告不存在（Phase 3产物）"
fi

# ============================================================
# 6. VERSION.txt检查
# ============================================================
echo ""
echo "--- 6. 版本信息 ---"

if [ -f "$SCRIPT_DIR/VERSION.txt" ]; then
    check_pass "VERSION.txt 存在"
    echo ""
    cat "$SCRIPT_DIR/VERSION.txt"
    echo ""
else
    check_warn "VERSION.txt 不存在"
fi

# ============================================================
# 7. Python环境检查（如果venv可用）
# ============================================================
echo ""
echo "--- 7. Python环境检查 ---"

VENV_PYTHON=""
for p in "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/../../venv/bin/python" "python3"; do
    if [ -x "$p" ] || command -v "$p" &>/dev/null; then
        VENV_PYTHON="$p"
        break
    fi
done

if [ -n "$VENV_PYTHON" ]; then
    PY_VER=$($VENV_PYTHON --version 2>&1)
    check_pass "Python: $PY_VER"

    # Check key packages
    if $VENV_PYTHON -c "import lightgbm" 2>/dev/null; then
        LGB_VER=$($VENV_PYTHON -c "import lightgbm; print(lightgbm.__version__)" 2>/dev/null || echo "unknown")
        check_pass "lightgbm: $LGB_VER"
    else
        check_fail "lightgbm 未安装"
    fi

    if $VENV_PYTHON -c "import optuna" 2>/dev/null; then
        OPT_VER=$($VENV_PYTHON -c "import optuna; print(optuna.__version__)" 2>/dev/null || echo "unknown")
        check_pass "optuna: $OPT_VER"
    else
        check_warn "optuna 未安装（训练阶段需要）"
    fi

    if $VENV_PYTHON -c "import qlib" 2>/dev/null; then
        QL_VER=$($VENV_PYTHON -c "import qlib; print(qlib.__version__)" 2>/dev/null || echo "unknown")
        check_pass "qlib: $QL_VER"
    else
        check_fail "qlib 未安装"
    fi

    if $VENV_PYTHON -c "import pandas; import numpy; import tables" 2>/dev/null; then
        check_pass "pandas/numpy/tables 已安装"
    else
        check_fail "pandas/numpy/tables 未完全安装"
    fi
else
    check_warn "Python环境不可用，跳过包检查"
fi

# ============================================================
# 8. 模型加载测试（如果环境完整）
# ============================================================
echo ""
echo "--- 8. 模型加载测试 ---"

if [ -f "$SCRIPT_DIR/models/quantaalpha_model.txt" ] && [ -n "$VENV_PYTHON" ]; then
    LOAD_RESULT=$($VENV_PYTHON -c "
import lightgbm as lgb
try:
    model = lgb.Booster(model_file='$SCRIPT_DIR/models/quantaalpha_model.txt')
    print(f'OK trees={model.num_trees()}')
except Exception as e:
    print(f'FAIL {e}')
" 2>&1)

    if [[ "$LOAD_RESULT" == OK* ]]; then
        check_pass ".txt模型加载成功 ($LOAD_RESULT)"
    else
        check_fail ".txt模型加载失败: $LOAD_RESULT"
    fi
else
    check_warn "跳过模型加载测试（需要模型文件+Python环境）"
fi

# ============================================================
# 总结
# ============================================================
echo ""
echo "======================================================================"
echo "验证结果"
echo "======================================================================"
echo -e "  ${GREEN}PASS: $PASS${NC}"
echo -e "  ${RED}FAIL: $FAIL${NC}"
echo -e "  ${YELLOW}WARN: $WARN${NC}"
echo ""

if [ $FAIL -gt 0 ]; then
    echo -e "${RED}❌ 验证失败：$FAIL 项未通过${NC}"
    exit 1
else
    echo -e "${GREEN}✅ 验证通过！所有关键检查项均已满足${NC}"
    exit 0
fi
