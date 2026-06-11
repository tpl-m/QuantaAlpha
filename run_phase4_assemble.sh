#!/bin/bash
# Phase 4: 生产版本组装 — QuantaAlpha v2.0
# 在Phase 1-3完成后运行，创建生产版本快照

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROD_DIR="$SCRIPT_DIR/quantaalpha_production_v2.0"

echo "======================================================================"
echo "Phase 4: 生产版本组装 (QuantaAlpha v2.0)"
echo "======================================================================"

# ============================================================
# 前置检查
# ============================================================
echo ""
echo "前置检查..."

# Check Optuna results exist
if [ ! -f "$SCRIPT_DIR/optuna_search_results/best_configuration.json" ]; then
    echo "❌ Optuna搜索结果不存在，请先运行 Phase 2"
    echo "   python3 train_model_optuna.py"
    exit 1
fi

# Check model file
if [ ! -f "$SCRIPT_DIR/exported_models/quantaalpha_model.txt" ]; then
    echo "❌ 模型文件不存在，请先运行 Phase 3"
    echo "   python3 train_model_simple.py --mode v5.0"
    exit 1
fi

# Consistency check: model metadata must match Optuna config
echo "  一致性检查..."
MODEL_FACTOR_COUNT=$(python3 -c "
import json
try:
    meta = open('$SCRIPT_DIR/exported_models/model_metadata.txt').read()
    # Extract factor count from metadata header line
    for line in meta.split('\n'):
        if '因子' in line and '模型' in line:
            import re
            m = re.search(r'(\d+)因子', line)
            if m: print(int(m.group(1))); break
    else: print(-1)
except: print(-1)
" 2>/dev/null)
OPTUNA_FACTOR_COUNT=$(python3 -c "import json; c=json.loads(open('$SCRIPT_DIR/optuna_search_results/best_configuration.json').read()); print(c['n_factors'])" 2>/dev/null)

if [ "$MODEL_FACTOR_COUNT" != "$OPTUNA_FACTOR_COUNT" ] && [ "$MODEL_FACTOR_COUNT" != "-1" ]; then
    echo "  ❌ 因子数不一致: 模型=$MODEL_FACTOR_COUNT, Optuna=$OPTUNA_FACTOR_COUNT"
    echo "  请先用 v5.0 模式重新训练: python3 train_model_simple.py --mode v5.0"
    exit 1
else
    echo "  ✅ 因子数一致: $OPTUNA_FACTOR_COUNT"
fi

echo "  ✅ 所有前置条件满足"

# ============================================================
# 创建目录结构
# ============================================================
echo ""
echo "创建目录结构..."

rm -rf "$PROD_DIR"
mkdir -p "$PROD_DIR/models"
mkdir -p "$PROD_DIR/algorithm"
mkdir -p "$PROD_DIR/joinquant_scripts"
mkdir -p "$PROD_DIR/docs"

echo "  ✅ 目录创建完成"

# ============================================================
# 复制模型文件
# ============================================================
echo ""
echo "复制模型文件..."

cp "$SCRIPT_DIR/exported_models/quantaalpha_model.txt" "$PROD_DIR/models/"
echo "  ✅ quantaalpha_model.txt"

if [ -f "$SCRIPT_DIR/exported_models/model_metadata.txt" ]; then
    cp "$SCRIPT_DIR/exported_models/model_metadata.txt" "$PROD_DIR/models/"
    echo "  ✅ model_metadata.txt"
fi

if [ -f "$SCRIPT_DIR/exported_models/quantaalpha_model.pkl" ]; then
    cp "$SCRIPT_DIR/exported_models/quantaalpha_model.pkl" "$PROD_DIR/models/"
    echo "  ✅ quantaalpha_model.pkl"
fi

# ============================================================
# 复制算法文件
# ============================================================
echo ""
echo "复制算法文件..."

cp "$SCRIPT_DIR/train_model_simple.py" "$PROD_DIR/algorithm/"
echo "  ✅ train_model_simple.py (含v5.0模式)"

cp "$SCRIPT_DIR/train_model_optuna.py" "$PROD_DIR/algorithm/"
echo "  ✅ train_model_optuna.py"

cp "$SCRIPT_DIR/compute_feature_cache.py" "$PROD_DIR/algorithm/"
echo "  ✅ compute_feature_cache.py"

if [ -f "$SCRIPT_DIR/smoke_test.py" ]; then
    cp "$SCRIPT_DIR/smoke_test.py" "$PROD_DIR/algorithm/"
    echo "  ✅ smoke_test.py"
fi

if [ -f "$SCRIPT_DIR/evaluate_model_comprehensive.py" ]; then
    cp "$SCRIPT_DIR/evaluate_model_comprehensive.py" "$PROD_DIR/algorithm/"
    echo "  ✅ evaluate_model_comprehensive.py"
fi

# ============================================================
# 复制聚宽脚本
# ⚠️  检查是否匹配v5.0模式：如果有optuna搜索结果，则只复制如果已有v5.0 loader
echo ""
echo "复制聚宽脚本..."

# 检查是否为v5.0包（有optuna搜索结果 = v5.0）
IS_V5=false
if [ -f "$SCRIPT_DIR/optuna_search_results/best_configuration.json" ]; then
    IS_V5=true
fi

if [ "$IS_V5" = true ]; then
    # v5.0包：不复制v4.0 loader（因子不匹配），添加说明文件
    echo "  ⚠️  v5.0包：跳过v4.0 loader复制（因子表达式不匹配）"
    echo "  v5.0模式需要手动实现因子计算函数" > "$PROD_DIR/joinquant_scripts/README_v5.txt"
    echo "  参考v4.0实现: quantaalpha_production_v1.1_fresh/joinquant_scripts/load_model_joinquant.py" >> "$PROD_DIR/joinquant_scripts/README_v5.txt"
    echo "  ✅ README_v5.txt (v5.0 JoinQuant说明)"
else
    # 非v5.0包：复制v4.0 loader
    if [ -f "$SCRIPT_DIR/exported_models/load_model_joinquant.py" ]; then
        cp "$SCRIPT_DIR/exported_models/load_model_joinquant.py" "$PROD_DIR/joinquant_scripts/"
        echo "  ✅ load_model_joinquant.py (v4.0)"
    fi
fi

# ============================================================
# 复制文档
# ============================================================
echo ""
echo "复制文档..."

if [ -f "$SCRIPT_DIR/factor_candidate_pool.json" ]; then
    cp "$SCRIPT_DIR/factor_candidate_pool.json" "$PROD_DIR/docs/"
    echo "  ✅ factor_candidate_pool.json"
fi

if [ -f "$SCRIPT_DIR/optuna_search_results/best_configuration.json" ]; then
    cp "$SCRIPT_DIR/optuna_search_results/best_configuration.json" "$PROD_DIR/docs/"
    echo "  ✅ best_configuration.json"
fi

if [ -f "$SCRIPT_DIR/optuna_search_results/all_trials.json" ]; then
    cp "$SCRIPT_DIR/optuna_search_results/all_trials.json" "$PROD_DIR/docs/"
    echo "  ✅ all_trials.json"
fi

# Copy evaluation reports
for f in "$SCRIPT_DIR/evaluation_reports/eval_report_"*.json; do
    if [ -f "$f" ]; then
        cp "$f" "$PROD_DIR/docs/"
        echo "  ✅ $(basename $f)"
    fi
done

# Copy selected_factors_config.json (needed by scripts)
if [ -f "$SCRIPT_DIR/evaluation_reports/selected_factors_config.json" ]; then
    cp "$SCRIPT_DIR/evaluation_reports/selected_factors_config.json" "$PROD_DIR/docs/"
    echo "  ✅ selected_factors_config.json"
fi

# Copy factor_candidate_pool.json (Phase 1 output, needed by analysis scripts)
if [ -f "$SCRIPT_DIR/factor_candidate_pool.json" ]; then
    cp "$SCRIPT_DIR/factor_candidate_pool.json" "$PROD_DIR/docs/"
    echo "  ✅ factor_candidate_pool.json"
fi

# Copy full optuna_search_results/ directory (needed by v5.0 mode)
if [ -d "$SCRIPT_DIR/optuna_search_results" ]; then
    cp -r "$SCRIPT_DIR/optuna_search_results" "$PROD_DIR/docs/optuna_search_results"
    echo "  ✅ optuna_search_results/ (complete)"
fi

# ============================================================
# 创建VERSION.txt
# ============================================================
echo ""
echo "创建VERSION.txt..."

# Get info from Optuna results
TRIAL_NUM=$(python3 -c "import json; c=json.loads(open('$SCRIPT_DIR/optuna_search_results/best_configuration.json').read()); print(c['trial_number'])")
SCORE=$(python3 -c "import json; c=json.loads(open('$SCRIPT_DIR/optuna_search_results/best_configuration.json').read()); print(f\"{c['score']:.6f}\")")
N_FACTORS=$(python3 -c "import json; c=json.loads(open('$SCRIPT_DIR/optuna_search_results/best_configuration.json').read()); print(c['n_factors'])")
SELECTED=$(python3 -c "import json; c=json.loads(open('$SCRIPT_DIR/optuna_search_results/best_configuration.json').read()); print(', '.join(c['selected_factors'][:5]))")
DATE=$(date '+%Y-%m-%d')
TREES=$(python3 -c "
import lightgbm as lgb
m = lgb.Booster(model_file='$SCRIPT_DIR/exported_models/quantaalpha_model.txt')
print(m.num_trees())
" 2>/dev/null || echo "N/A")

cat > "$PROD_DIR/VERSION.txt" << EOF
QuantaAlpha v2.0 - Optuna联合搜索版本
=======================================
创建日期: $DATE
来源: Optuna Trial #$TRIAL_NUM
Score: $SCORE

因子数: $N_FACTORS
选中因子: $SELECTED

模型树数: $TREES
模型格式: LightGBM .txt (JoinQuant兼容)

搜索配置:
  - 算法: Optuna TPE
  - Trials: 200
  - 因子约束: 3-15个
  - 目标函数: mean_ic + ICIR门控 + 回撤惩罚 + IC门控

质量门控阈值:
  - 日截面IC > 0.015
  - ICIR > 0.12
  - 年化收益 > 5%
  - 最大回撤 > -15%
  - 胜率 > 52%
  - IC>0比例 > 55%

文件清单:
  models/quantaalpha_model.txt   - LightGBM模型（主要生产格式）
  models/model_metadata.txt      - 模型元信息
  algorithm/train_model_simple.py - 训练脚本（含v5.0模式）
  algorithm/train_model_optuna.py - Optuna搜索脚本
  algorithm/compute_feature_cache.py - 特征缓存脚本
  joinquant_scripts/load_model_joinquant.py - 聚宽加载脚本
  docs/best_configuration.json   - 最优配置
  docs/factor_candidate_pool.json - 候选因子池
  docs/all_trials.json           - 所有搜索结果
EOF

echo "  ✅ VERSION.txt 创建完成"

# ============================================================
# 运行验证
# ============================================================
echo ""
echo "运行完整性验证..."

if [ -f "$SCRIPT_DIR/verify_v2_package.sh" ]; then
    # Copy verifier into the package and run it there (verifies $PROD_DIR contents)
    cp "$SCRIPT_DIR/verify_v2_package.sh" "$PROD_DIR/verify_package.sh"
    chmod +x "$PROD_DIR/verify_package.sh"
    bash "$PROD_DIR/verify_package.sh"
else
    echo "  ⚠️  verify_v2_package.sh 不存在，跳过验证"
fi

# ============================================================
# 总结
# ============================================================
echo ""
echo "======================================================================"
echo "✅ Phase 4 完成！"
echo "======================================================================"
echo ""
echo "📦 生产版本目录: $PROD_DIR"
echo ""
echo "📋 后续步骤:"
echo "  1. 运行验证: bash $PROD_DIR/../verify_v2_package.sh"
echo "  2. 上传模型: 将 quantaalpha_model.txt 上传到聚宽平台"
echo "  3. 部署脚本: 复制 load_model_joinquant.py 到聚宽策略"
echo "  4. 运行回测: 在聚宽平台回测验证"
echo ""
echo "⚠️  注意:"
echo "  - 此目录为不可变快照（v2.0），任何修改必须创建新版本"
echo "  - v5.0聚宽脚本中的因子计算需根据实际Qlib表达式手动实现"
