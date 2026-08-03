#!/bin/bash
echo "=== 环境检查 ==="
echo "1. Conda环境:"
conda info --envs 2>/dev/null | grep "*" || echo "未使用conda"
echo "当前CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo ""

echo "2. Python路径:"
which python
python --version
echo ""

echo "3. Pip路径:"
which pip
pip --version
echo ""

echo "4. 版本对比:"
echo "通过pip list:"
pip list 2>/dev/null | grep -E "transformers|accelerate|torch" || echo "pip不可用"
echo ""
echo "通过python -m pip list:"
python -m pip list 2>/dev/null | grep -E "transformers|accelerate|torch" || echo "python -m pip不可用"
echo ""
echo "通过python导入:"
python -c "
try:
    import transformers, accelerate, torch
    print(f'Transformers: {transformers.__version__}')
    print(f'Accelerate: {accelerate.__version__}')
    print(f'Torch: {torch.__version__}')
except Exception as e:
    print(f'导入错误: {e}')
" 2>/dev/null
echo ""
echo "5. Site-packages路径:"
python -c "import sys; print('\n'.join([p for p in sys.path if 'site-packages' in p]))" 2>/dev/null