"""
evaluate_gemini_3.0_pro.py - Gemini 3.0 Proを用いたRAGの評価スクリプト

evaluate_gemini_base.pyの基底クラスを継承し、
Gemini 3.0 Proモデルを使用してRAGの評価を実行する。
"""

from evaluate_gemini_base import GeminiRAGEvaluator

# モデル設定
MODEL_NAME = "gemini-3-pro-preview"


class Gemini30ProEvaluator(GeminiRAGEvaluator):
    """Gemini 3.0 Proを用いたRAGの評価クラス"""
    def __init__(self):
        super().__init__(model_name=MODEL_NAME)


def main():
    """メイン関数"""
    evaluator = Gemini30ProEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
