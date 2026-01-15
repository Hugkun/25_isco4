"""
evaluate_gemini_2.5_flash_lite.py - Gemini 2.5 Flash Liteを用いたRAGの評価スクリプト

evaluate_gemini_base.pyの基底クラスを継承し、
Gemini 2.5 Flash Liteモデルを使用してRAGの評価を実行する。
"""

from evaluate_gemini_base import GeminiRAGEvaluator

# モデル設定
MODEL_NAME = "gemini-2.5-flash-lite"


class Gemini25FlashLiteEvaluator(GeminiRAGEvaluator):
    """Gemini 2.5 Flash Liteを用いたRAGの評価クラス"""

    def __init__(self):
        super().__init__(model_name=MODEL_NAME)


def main():
    """メイン関数"""
    evaluator = Gemini25FlashLiteEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
