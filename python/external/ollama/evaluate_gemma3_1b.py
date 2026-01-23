"""
evaluate_gemma3_1b.py - gemma3:1bを用いたRAGの評価スクリプト

evaluate_ollama_base.pyの基底クラスを継承し、
gemma3:1bモデルを使用してRAGの評価を実行する。
"""

import argparse

from evaluate_ollama_base import (
    DEFAULT_EVALUATOR_EMBEDDING_MODEL,
    DEFAULT_EVALUATOR_LLM,
    DEFAULT_RAG_EMBEDDING_MODEL,
    OllamaRAGEvaluator,
)

# モデル設定
MODEL_NAME = "gemma3:1b"


class Gemma31BEvaluator(OllamaRAGEvaluator):
    """gemma3:1bを用いたRAGの評価クラス"""

    def __init__(
        self,
        rag_embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
        evaluator_llm: str = DEFAULT_EVALUATOR_LLM,
        evaluator_embedding_model: str = DEFAULT_EVALUATOR_EMBEDDING_MODEL,
    ):
        """
        Args:
            rag_embedding_model: RAG用埋め込みモデルの名前（デフォルト: embeddinggemma:latest）
            evaluator_llm: 評価用LLMの名前（デフォルト: gemma3:12b）
            evaluator_embedding_model: 評価用埋め込みモデルの名前（デフォルト: embeddinggemma:latest）
        """
        super().__init__(
            model_name=MODEL_NAME,
            rag_embedding_model=rag_embedding_model,
            evaluator_llm=evaluator_llm,
            evaluator_embedding_model=evaluator_embedding_model,
        )


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description=f"{MODEL_NAME}を用いたRAGの構築・評価を実行する"
    )
    parser.add_argument(
        "--rag-embedding-model",
        type=str,
        default=DEFAULT_RAG_EMBEDDING_MODEL,
        help=f"RAG用埋め込みモデルの名前（デフォルト: {DEFAULT_RAG_EMBEDDING_MODEL}）",
    )
    parser.add_argument(
        "--evaluator-llm",
        type=str,
        default=DEFAULT_EVALUATOR_LLM,
        help=f"評価用LLMの名前（デフォルト: {DEFAULT_EVALUATOR_LLM}）",
    )
    parser.add_argument(
        "--evaluator-embedding-model",
        type=str,
        default=DEFAULT_EVALUATOR_EMBEDDING_MODEL,
        help=f"評価用埋め込みモデルの名前（デフォルト: {DEFAULT_EVALUATOR_EMBEDDING_MODEL}）",
    )
    return parser.parse_args()


def main():
    """メイン関数"""
    args = parse_args()

    evaluator = Gemma31BEvaluator(
        rag_embedding_model=args.rag_embedding_model,
        evaluator_llm=args.evaluator_llm,
        evaluator_embedding_model=args.evaluator_embedding_model,
    )
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
