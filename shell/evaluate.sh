#!/bin/bash
#
# evaluate.sh - RAGの評価を実行するスクリプト
#
# 使用方法:
#   ./evaluate.sh                     # 全てのモデルを評価
#   ./evaluate.sh <model_name>        # 指定したモデルのみ評価
#
# 対応モデル:
#   - gemini-2.5-flash-lite
#   - gemini-2.5-flash
#   - gemini-2.5-pro
#   - gemini-3-flash-preview
#   - gemini-3-pro-preview
#

# スクリプトのディレクトリを取得
SCRIPT_DIR=$(dirname "$0")
GEMINI_DIR="$SCRIPT_DIR/../python/external/gemini"

# モデル名からスクリプト名を取得する関数
get_script_name() {
    local model_name=$1
    case "$model_name" in
        "gemini-2.5-flash-lite")
            echo "evaluate_gemini_2.5_flash_lite.py"
            ;;
        "gemini-2.5-flash")
            echo "evaluate_gemini_2.5_flash.py"
            ;;
        "gemini-2.5-pro")
            echo "evaluate_gemini_2.5_pro.py"
            ;;
        "gemini-3-flash-preview")
            echo "evaluate_gemini_3.0_flash.py"
            ;;
        "gemini-3-pro-preview")
            echo "evaluate_gemini_3.0_pro.py"
            ;;
        *)
            echo ""
            ;;
    esac
}

# 評価スクリプトを実行する関数
run_evaluation() {
    local script_name=$1
    echo "=== $script_name を実行中 ==="
    (cd "$GEMINI_DIR" && uv run python "$script_name")
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
    local scripts=(
        "evaluate_gemini_2.5_flash_lite.py"
        "evaluate_gemini_2.5_flash.py"
        "evaluate_gemini_2.5_pro.py"
        "evaluate_gemini_3.0_flash.py"
        "evaluate_gemini_3.0_pro.py"
    )

    for script in "${scripts[@]}"; do
        run_evaluation "$script"
        if [ $? -ne 0 ]; then
            echo "評価処理を中断しました"
            exit 1
        fi
    done
}

# 対応モデル一覧を表示する関数
show_available_models() {
    echo "対応モデル一覧:"
    echo "  - gemini-2.5-flash-lite"
    echo "  - gemini-2.5-flash"
    echo "  - gemini-2.5-pro"
    echo "  - gemini-3-flash-preview"
    echo "  - gemini-3-pro-preview"
}

# メイン処理
main() {
    if [ $# -eq 0 ]; then
        # 引数がない場合は全てのスクリプトを実行
        echo "全てのモデルを評価します..."
        echo ""
        run_all_evaluations
        echo "全ての評価が完了しました"
    else
        # 引数がある場合は指定されたモデルのみ評価
        local model_name=$1
        local script
        script=$(get_script_name "$model_name")

        if [ -z "$script" ]; then
            echo "エラー: 不明なモデル名です: $model_name"
            echo ""
            show_available_models
            exit 1
        fi

        echo "モデル '$model_name' を評価します..."
        echo ""
        run_evaluation "$script"
        if [ $? -ne 0 ]; then
            exit 1
        fi
        echo "評価が完了しました"
    fi
}

main "$@"
