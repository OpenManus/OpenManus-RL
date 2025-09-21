#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Together AI 性能测试脚本"
echo "=============================="

# 检查环境变量
if [ -z "${TOGETHER_API_KEY:-}" ]; then
    echo "❌ 错误: 未设置 TOGETHER_API_KEY 环境变量"
    echo "请运行: export TOGETHER_API_KEY=your_api_key_here"
    exit 1
fi

echo "✅ TOGETHER_API_KEY 已设置"

# 进入测试目录
cd "$(dirname "$0")"

# 给脚本添加执行权限
chmod +x test_togetherai.py

echo ""
echo "选择测试模式:"
echo "1) 快速测试 (10次请求, 3并行)"
echo "2) 标准测试 (100次请求, 10并行)"
echo "3) 自定义测试"
echo "4) 压力测试 (200次请求, 20并行)"

read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo "🏃 运行快速测试..."
        python test_togetherai.py --quick
        ;;
    2)
        echo "📊 运行标准测试..."
        python test_togetherai.py --requests 100 --parallel 10
        ;;
    3)
        read -p "输入请求数量: " requests
        read -p "输入并行数量: " parallel
        echo "🔧 运行自定义测试..."
        python test_togetherai.py --requests "$requests" --parallel "$parallel"
        ;;
    4)
        echo "💪 运行压力测试..."
        python test_togetherai.py --requests 200 --parallel 20
        ;;
    *)
        echo "❌ 无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "🎉 测试完成！"
