#!/bin/bash
#
# evaluate.sh - RAGの構築・評価を行うPythonスクリプトを一括で実行
#
# 使用方法:
#   ./evaluate.sh [オプション]
#
# オプション:
#   --llm <model_name>                RAG用LLMの名前（指定しない場合は全モデルを評価）
#   --rag-embedding-model <model>     RAG用埋め込みモデルの名前
#   --evaluator-llm <model>           評価用LLMの名前
#   --evaluator-embedding-model <model> 評価用埋め込みモデルの名前
#
# 対応RAG用LLM:
#   Gemini:
#     - gemini-2.5-flash-lite
#     - gemini-2.5-flash
#     - gemini-2.5-pro
#     - gemini-3-flash-preview
#     - gemini-3-pro-preview
#   Ollama:
#     - gemma3:12b
#     - gpt-oss:20b
#

set -e

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GEMINI_DIR="$PROJECT_ROOT/python/external/gemini"
OLLAMA_DIR="$PROJECT_ROOT/python/external/ollama"

# コマンドライン引数
RAG_LLM=""
RAG_EMBEDDING_MODEL=""
EVALUATOR_LLM=""
EVALUATOR_EMBEDDING_MODEL=""

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --llm)
            RAG_LLM="$2"
            shift 2
            ;;
        --rag-embedding-model)
            RAG_EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --evaluator-llm)
            EVALUATOR_LLM="$2"
            shift 2
            ;;
        --evaluator-embedding-model)
            EVALUATOR_EMBEDDING_MODEL="$2"
            shift 2
            ;;
        *)
            echo "不明な引数: $1"
            exit 1
            ;;
    esac
done

# Pythonスクリプトに渡す引数を構築する関数
build_python_args() {
    local args=""
    if [ -n "$RAG_EMBEDDING_MODEL" ]; then
        args="$args --rag-embedding-model $RAG_EMBEDDING_MODEL"
    fi
    if [ -n "$EVALUATOR_LLM" ]; then
        args="$args --evaluator-llm $EVALUATOR_LLM"
    fi
    if [ -n "$EVALUATOR_EMBEDDING_MODEL" ]; then
        args="$args --evaluator-embedding-model $EVALUATOR_EMBEDDING_MODEL"
    fi
    echo "$args"
}

# モデル名からスクリプト情報を取得する関数（ディレクトリとスクリプト名を返す）
get_script_info() {
    local model_name=$1
    case "$model_name" in
        "gemini-2.5-flash-lite")
            echo "$GEMINI_DIR evaluate_gemini_2.5_flash_lite.py"
            ;;
        "gemini-2.5-flash")
            echo "$GEMINI_DIR evaluate_gemini_2.5_flash.py"
            ;;
        "gemini-2.5-pro")
            echo "$GEMINI_DIR evaluate_gemini_2.5_pro.py"
            ;;
        "gemini-3-flash-preview")
            echo "$GEMINI_DIR evaluate_gemini_3.0_flash.py"
            ;;
        "gemini-3-pro-preview")
            echo "$GEMINI_DIR evaluate_gemini_3.0_pro.py"
            ;;
        "gemma3:1b")
            echo "$OLLAMA_DIR evaluate_gemma3_1b.py"
            ;;
        "gemma3:4b")
            echo "$OLLAMA_DIR evaluate_gemma3_4b.py"
            ;;
        "gemma3:12b")
            echo "$OLLAMA_DIR evaluate_gemma3_12b.py"
            ;;
        "gemma3:27b")
            echo "$OLLAMA_DIR evaluate_gemma3_27b.py"
            ;;
        "gpt-oss:20b")
            echo "$OLLAMA_DIR evaluate_gpt-oss_20b.py"
            ;;
        "gpt-oss:120b")
            echo "$OLLAMA_DIR evaluate_gpt-oss_120b.py"
            ;;
        *)
            echo ""
            ;;
    esac
}

# 評価スクリプトを実行する関数
run_evaluation() {
    local script_dir=$1
    local script_name=$2
    local python_args
    python_args=$(build_python_args)

    echo "=== $script_name を実行中 ==="
    if [ -n "$python_args" ]; then
        echo "  引数: $python_args"
    fi

    (cd "$script_dir" && uv run python "$script_name" $python_args)

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "エラー: $script_name の実行に失敗しました (exit code: $exit_code)"
        return $exit_code
    fi
    echo "=== $script_name の実行完了 ==="
    echo ""
}

# 全てのスクリプトを実行する関数
run_all_evaluations() {
    # Geminiモデル
    local gemini_scripts=(
        "evaluate_gemini_2.5_flash_lite.py"
        "evaluate_gemini_2.5_flash.py"
        "evaluate_gemini_2.5_pro.py"
        "evaluate_gemini_3.0_flash.py"
        "evaluate_gemini_3.0_pro.py"
    )

    # Ollamaモデル
    local ollama_scripts=(
        "evaluate_gemma3_12b.py"
        "evaluate_gpt-oss_20b.py"
    )

    # Geminiモデルの評価
    for script in "${gemini_scripts[@]}"; do
        run_evaluation "$GEMINI_DIR" "$script"
        if [ $? -ne 0 ]; then
            echo "評価処理を中断しました"
            exit 1
        fi
    done

    # Ollamaモデルの評価
    for script in "${ollama_scripts[@]}"; do
        run_evaluation "$OLLAMA_DIR" "$script"
        if [ $? -ne 0 ]; then
            echo "評価処理を中断しました"
            exit 1
        fi
    done
}

# 対応モデル一覧を表示する関数
show_available_models() {
    echo "対応RAG用LLM一覧:"
    echo "  Gemini:"
    echo "    - gemini-2.5-flash-lite"
    echo "    - gemini-2.5-flash"
    echo "    - gemini-2.5-pro"
    echo "    - gemini-3-flash-preview"
    echo "    - gemini-3-pro-preview"
    echo "  Ollama:"
    echo "    - gemma3:1b"
    echo "    - gemma3:4b"
    echo "    - gemma3:12b"
    echo "    - gemma3:27b"
    echo "    - gpt-oss:20b"
    echo "    - gpt-oss:120b"
}

# メイン処理
main() {
    echo "=== RAG評価スクリプト ==="

    if [ -z "$RAG_LLM" ]; then
        # RAG用LLMが指定されていない場合は全てのスクリプトを実行
        echo "全てのモデルを評価します..."
        echo ""
        run_all_evaluations
        echo "全ての評価が完了しました"
    else
        # RAG用LLMが指定されている場合は対応するスクリプトのみ実行
        local script_info
        script_info=$(get_script_info "$RAG_LLM")

        if [ -z "$script_info" ]; then
            echo "エラー: 不明なモデル名です: $RAG_LLM"
            echo ""
            show_available_models
            exit 1
        fi

        local script_dir
        local script_name
        script_dir=$(echo "$script_info" | awk '{print $1}')
        script_name=$(echo "$script_info" | awk '{print $2}')

        echo "モデル '$RAG_LLM' を評価します..."
        echo ""
        run_evaluation "$script_dir" "$script_name"
        if [ $? -ne 0 ]; then
            exit 1
        fi
        echo "評価が完了しました"
    fi
}

main
