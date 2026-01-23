"""
evaluate_ollama_base.py - Ollamaモデルを用いたRAGの評価用基底クラス

outputs/faiss.indexを読み込んでRAGを構築し、
Ragasを用いて評価を実行する基底クラスを定義する。
"""

import os
import platform
import time
from abc import ABC
from pathlib import Path

import psutil

import pandas as pd
from dotenv import load_dotenv
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper, llm_factory
from ragas.metrics import (
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
    ResponseRelevancy,
)

# .envファイルの読み込み
load_dotenv()

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FAISS_INDEX_PATH = PROJECT_ROOT / "outputs" / "faiss.index"
VALID_DATA_PATH = PROJECT_ROOT / "data" / "valid.csv"
RESULTS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external" / "ollama"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "summary.xlsx"

# デフォルト設定
DEFAULT_RAG_EMBEDDING_MODEL = "embeddinggemma:latest"
DEFAULT_EVALUATOR_LLM = "gemma3:12b"
DEFAULT_EVALUATOR_EMBEDDING_MODEL = "embeddinggemma:latest"

# Gemini APIの埋め込みモデル名
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# Gemini LLMモデル名のリスト
GEMINI_LLM_MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

import sys
python_path = PROJECT_ROOT / "python"
sys.path.insert(0, str(python_path))
from utils.hardware_info import get_hardware_info


class OllamaRAGEvaluator(ABC):
    """Ollamaモデルを用いたRAGの評価用基底クラス"""

    def __init__(
        self,
        model_name: str,
        rag_embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
        evaluator_llm: str = DEFAULT_EVALUATOR_LLM,
        evaluator_embedding_model: str = DEFAULT_EVALUATOR_EMBEDDING_MODEL,
    ):
        """
        Args:
            model_name: RAGに使用するOllamaモデルの名前
            rag_embedding_model: RAG用埋め込みモデルの名前（デフォルト: embeddinggemma:latest）
            evaluator_llm: 評価用LLMの名前（デフォルト: gemma3:12b）
            evaluator_embedding_model: 評価用埋め込みモデルの名前（デフォルト: embeddinggemma:latest）
        """
        self.model_name = model_name
        self.rag_embedding_model = rag_embedding_model
        self.evaluator_llm = evaluator_llm
        self.evaluator_embedding_model = evaluator_embedding_model

        # Gemini API用のキー（評価用LLMや埋め込みモデルで使用する可能性がある）
        self._api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        self._llm = None
        self._vector_store = None
        self._retriever = None

    def _initialize_llm(self) -> ChatOllama:
        """RAG用LLMを初期化する"""
        return ChatOllama(model=self.model_name)

    def _initialize_embeddings(
        self,
    ) -> GoogleGenerativeAIEmbeddings | OllamaEmbeddings:
        """RAG用埋め込みモデルを初期化する"""
        if self.rag_embedding_model == GEMINI_EMBEDDING_MODEL:
            if not self._api_key:
                raise ValueError(
                    "GOOGLE_API_KEY または GEMINI_API_KEY が設定されていません。.envファイルを確認してください。"
                )
            return GoogleGenerativeAIEmbeddings(
                model=f"models/{GEMINI_EMBEDDING_MODEL}",
                google_api_key=self._api_key,
            )
        else:
            return OllamaEmbeddings(model=self.rag_embedding_model)

    def _initialize_evaluator_llm(self) -> LangchainLLMWrapper:
        """評価用LLMを初期化する"""
        if self.evaluator_llm in GEMINI_LLM_MODELS:
            # Gemini APIを使用
            if not self._api_key:
                raise ValueError(
                    "GOOGLE_API_KEY または GEMINI_API_KEY が設定されていません。.envファイルを確認してください。"
                )
            client = genai.Client(api_key=self._api_key)
            return llm_factory(
                model=self.evaluator_llm,
                provider="google",
                client=client,
            )
        else:
            # Ollamaを使用
            llm = ChatOllama(model=self.evaluator_llm)
            return LangchainLLMWrapper(llm)

    def _initialize_evaluator_embeddings(
        self,
    ) -> GoogleGenerativeAIEmbeddings | OllamaEmbeddings:
        """評価用埋め込みモデルを初期化する"""
        if self.evaluator_embedding_model == GEMINI_EMBEDDING_MODEL:
            if not self._api_key:
                raise ValueError(
                    "GOOGLE_API_KEY または GEMINI_API_KEY が設定されていません。.envファイルを確認してください。"
                )
            return GoogleGenerativeAIEmbeddings(
                model=f"models/{GEMINI_EMBEDDING_MODEL}",
                google_api_key=self._api_key,
            )
        else:
            return OllamaEmbeddings(model=self.evaluator_embedding_model)

    def _load_vector_store(self) -> FAISS:
        """FAISSベクトルストアを読み込む"""
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISSインデックスが見つかりません: {FAISS_INDEX_PATH}"
            )

        embeddings = self._initialize_embeddings()
        return FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def _load_validation_data(self) -> pd.DataFrame:
        """検証用データを読み込む"""
        if not VALID_DATA_PATH.exists():
            raise FileNotFoundError(
                f"検証用データが見つかりません: {VALID_DATA_PATH}"
            )

        return pd.read_csv(VALID_DATA_PATH)

    def _build_rag_chain(self):
        """RAGチェーンを構築する"""
        self._llm = self._initialize_llm()
        self._vector_store = self._load_vector_store()
        self._retriever = self._vector_store.as_retriever(search_kwargs={"k": 4})

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "あなたは質問に回答するアシスタントです。"
                "以下のコンテキストを参考にして質問に回答してください。\n\n"
                "コンテキスト:\n{context}",
            ),
            ("human", "{question}"),
        ])

        return prompt

    def _query_llm(self, question: str) -> tuple[str, float]:
        """LLM単体で質問に回答し、実行時間を計測する"""
        start_time = time.time()
        response = self._llm.invoke(question)
        execution_time = time.time() - start_time

        # Ollamaの場合、contentは文字列
        return response.content, execution_time

    def _query_rag(
        self, question: str, prompt: ChatPromptTemplate
    ) -> tuple[str, str, float]:
        """RAGで質問に回答し、実行時間を計測する"""
        start_time = time.time()

        # コンテキストを検索
        docs = self._retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # LLMで回答を生成
        messages = prompt.format_messages(context=context, question=question)
        response = self._llm.invoke(messages)

        execution_time = time.time() - start_time

        # Ollamaの場合、contentは文字列
        return response.content, context, execution_time

    def _create_evaluation_dataset(
        self, results: list[dict]
    ) -> EvaluationDataset:
        """Ragas評価用データセットを作成する"""
        samples = []
        for result in results:
            sample = SingleTurnSample(
                user_input=result["question"],
                retrieved_contexts=[result["context"]],
                response=result["rag.answer"],
                reference=result["ground_truth"],
            )
            samples.append(sample)
        return EvaluationDataset(samples=samples)

    def _evaluate_with_ragas(
        self, dataset: EvaluationDataset
    ) -> dict:
        """Ragasで評価を実行する"""
        # 評価用LLMの初期化
        llm = self._initialize_evaluator_llm()

        # 評価用埋め込みモデルの初期化
        embeddings = self._initialize_evaluator_embeddings()

        # 評価指標の設定
        metrics = [
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
            ContextEntityRecall(llm=llm),
            NoiseSensitivity(llm=llm),
            ResponseRelevancy(llm=llm),
            Faithfulness(llm=llm),
        ]

        # 評価の実行
        evaluation_result = evaluate(
            dataset=dataset, metrics=metrics, embeddings=embeddings, run_config=RunConfig(timeout=3000.0)
        ) # タイムアウト3000秒
        return evaluation_result

    def _evaluate_llm_response_relevancy(
        self, results: list[dict]
    ) -> list[float]:
        """LLM回答のResponse Relevancyを評価する"""
        # 評価用LLMの初期化
        llm = self._initialize_evaluator_llm()

        # 評価用埋め込みモデルの初期化
        embeddings = self._initialize_evaluator_embeddings()

        samples = []
        for result in results:
            sample = SingleTurnSample(
                user_input=result["question"],
                retrieved_contexts=[""],
                response=result["llm.answer"],
                reference=result["ground_truth"],
            )
            samples.append(sample)

        dataset = EvaluationDataset(samples=samples)
        metric = ResponseRelevancy(llm=llm)
        evaluation_result = evaluate(
            dataset=dataset, metrics=[metric], embeddings=embeddings, run_config=RunConfig(timeout=3000.0)
        ) # タイムアウト3000秒

        return evaluation_result.to_pandas()["answer_relevancy"].tolist()

    def _save_results(
        self, results: list[dict], ragas_scores: pd.DataFrame
    ) -> None:
        """評価結果を保存する"""
        # results_{model_name}.xlsxの作成
        results_df = pd.DataFrame(results)

        # Ragasスコアを追加
        results_df["rag.context_precision"] = ragas_scores["context_precision"]
        results_df["rag.context_recall"] = ragas_scores["context_recall"]
        results_df["rag.context_entities_recall"] = ragas_scores[
            "context_entity_recall"
        ]
        results_df["rag.noise_sensitivity"] = ragas_scores[
            "noise_sensitivity(mode=relevant)"
        ]
        results_df["rag.response_relevancy"] = ragas_scores["answer_relevancy"]
        results_df["rag.faithfulness"] = ragas_scores["faithfulness"]

        # カラム順序を整理
        column_order = [
            "question",
            "context",
            "llm.execution_time",
            "rag.execution_time",
            "llm.answer",
            "rag.answer",
            "ground_truth",
            "llm.response_relevancy",
            "rag.context_precision",
            "rag.context_recall",
            "rag.context_entities_recall",
            "rag.noise_sensitivity",
            "rag.response_relevancy",
            "rag.faithfulness",
        ]
        results_df = results_df[column_order]

        # 結果ファイルの保存
        RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results_file = RESULTS_OUTPUT_DIR / f"results_{self.model_name}.xlsx"
        results_df.to_excel(results_file, index=False, engine="openpyxl")
        print(f"  結果ファイルを保存しました: {results_file}")

        # summary.xlsxの更新
        self._update_summary(results_df)

    def _format_time_hhmm(self, seconds: float) -> str:
        """秒数をhh:mm形式に変換する"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours:02d}:{minutes:02d}"

    def _update_summary(self, results_df: pd.DataFrame) -> None:
        """summary.xlsxを更新する"""
        # 合計実行時間を計算してhh:mm形式に変換
        llm_total_time = results_df["llm.execution_time"].sum()
        rag_total_time = results_df["rag.execution_time"].sum()

        # サマリー行を作成
        summary_row = {
            "llm": self.model_name,
            "embedding_model": self.rag_embedding_model,
            "eval_n": len(results_df),
            "hardware": get_hardware_info(),
            "llm.total_execution_time": self._format_time_hhmm(llm_total_time),
            "rag.total_execution_time": self._format_time_hhmm(rag_total_time),
            "llm.avg_response_relevancy": results_df["llm.response_relevancy"].mean(),
            "rag.avg_context_precision": results_df["rag.context_precision"].mean(),
            "rag.avg_context_recall": results_df["rag.context_recall"].mean(),
            "rag.avg_context_entities_recall": results_df[
                "rag.context_entities_recall"
            ].mean(),
            "rag.avg_noise_sensitivity": results_df["rag.noise_sensitivity"].mean(),
            "rag.avg_response_relevancy": results_df["rag.response_relevancy"].mean(),
            "rag.avg_faithfulness": results_df["rag.faithfulness"].mean(),
        }

        # 既存のsummary.xlsxを読み込む or 新規作成
        if SUMMARY_OUTPUT_PATH.exists():
            summary_df = pd.read_excel(SUMMARY_OUTPUT_PATH, engine="openpyxl")
            # 同じモデルの行が存在する場合は削除
            summary_df = summary_df[summary_df["llm"] != self.model_name]
        else:
            summary_df = pd.DataFrame()

        # 新しい行を追加
        new_row_df = pd.DataFrame([summary_row])
        summary_df = pd.concat([summary_df, new_row_df], ignore_index=True)

        # 保存
        summary_df.to_excel(SUMMARY_OUTPUT_PATH, index=False, engine="openpyxl")
        print(f"  サマリーファイルを更新しました: {SUMMARY_OUTPUT_PATH}")

    def run_evaluation(self) -> None:
        """評価を実行する"""
        print(f"=== Ollama RAG評価: {self.model_name} ===")

        # RAGチェーンの構築
        print("\n1. RAGチェーンの構築")
        prompt = self._build_rag_chain()
        print(f"  RAG用LLM: {self.model_name}")
        print(f"  RAG用埋め込みモデル: {self.rag_embedding_model}")
        print(f"  評価用LLM: {self.evaluator_llm}")
        print(f"  評価用埋め込みモデル: {self.evaluator_embedding_model}")

        # 検証用データの読み込み
        print("\n2. 検証用データの読み込み")
        valid_df = self._load_validation_data()
        print(f"  読み込み完了: {len(valid_df)}件の質問")

        # 各質問に対してLLMとRAGで回答を生成
        print("\n3. 回答の生成")
        results = []
        for idx, row in valid_df.iterrows():
            question = row["user_input"]
            ground_truth = row["reference"]

            print(f"  質問 {idx + 1}/{len(valid_df)}: {question[:30]}...")

            # LLM単体で回答
            llm_answer, llm_time = self._query_llm(question)

            # RAGで回答
            rag_answer, context, rag_time = self._query_rag(question, prompt)

            results.append({
                "question": question,
                "context": context,
                "llm.execution_time": llm_time,
                "rag.execution_time": rag_time,
                "llm.answer": llm_answer,
                "rag.answer": rag_answer,
                "ground_truth": ground_truth,
            })

        # LLMのResponse Relevancyを評価
        print("\n4. LLM回答のResponse Relevancy評価")
        llm_response_relevancy = self._evaluate_llm_response_relevancy(results)
        for i, score in enumerate(llm_response_relevancy):
            results[i]["llm.response_relevancy"] = score

        # RAGの評価
        print("\n5. RAGの評価 (Ragas)")
        dataset = self._create_evaluation_dataset(results)
        ragas_result = self._evaluate_with_ragas(dataset)
        ragas_scores = ragas_result.to_pandas()

        # 結果の保存
        print("\n6. 結果の保存")
        self._save_results(results, ragas_scores)

        print("\n=== 評価完了 ===")
