"""
evaluate_gemini_2.5_pro.py - Gemini 2.5 Proを用いたRAGの評価スクリプト

evaluate_gemini_base.pyの基底クラスを継承し、
Gemini 2.5 Proモデルを使用してRAGの評価を実行する。
"""

from evaluate_gemini_base import GeminiRAGEvaluator

# モデル設定
MODEL_NAME = "gemini-2.5-pro"


class Gemini25ProEvaluator(GeminiRAGEvaluator):
    """Gemini 2.5 Proを用いたRAGの評価クラス"""

    def __init__(self):
        super().__init__(model_name=MODEL_NAME)


def main():
    """メイン関数"""
    evaluator = Gemini25ProEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
