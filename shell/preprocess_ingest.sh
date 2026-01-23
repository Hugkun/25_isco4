#!/bin/bash
#
# preprocess_ingest.sh - データの前処理とベクトルデータベース作成を実行
#
# 使用方法:
#   ./preprocess_ingest.sh [--embedding-model <model_name>]
#
# 引数:
#   --embedding-model: RAG用埋め込みモデルの名前（省略可能、デフォルト: embeddinggemma:latest）
#

set -e

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# コマンドライン引数の解析
EMBEDDING_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        *)
            echo "不明な引数: $1"
            exit 1
            ;;
    esac
done

echo "=== データの前処理とベクトルデータベース作成 ==="

# 前処理の実行
echo ""
echo "--- preprocess.py の実行 ---"
uv run python "$PROJECT_ROOT/python/tools/preprocess.py"

# ベクトルデータベース作成の実行
echo ""
echo "--- ingest.py の実行 ---"
if [ -n "$EMBEDDING_MODEL" ]; then
    uv run python "$PROJECT_ROOT/python/tools/ingest.py" --embedding-model "$EMBEDDING_MODEL"
else
    uv run python "$PROJECT_ROOT/python/tools/ingest.py"
fi

echo ""
echo "=== 処理完了 ==="