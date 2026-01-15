"""
evaluate_gemini_base.py - Geminiを用いたRAGの評価用基底クラス

outputs/faiss.indexを読み込んでRAGを構築し、
Ragasを用いて評価を実行する基底クラスを定義する。
"""

import os
import time
from abc import ABC
from pathlib import Path
import re

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from google import genai
from ragas.llms import llm_factory, InstructorBaseRagasLLM
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
RESULTS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external" / "gemini"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "summary.xlsx"

# 評価用LLM設定
EVALUATOR_MODEL = "gemini-2.5-flash-lite"

# 評価用埋め込みモデル設定
EVALUATOR_EMBEDDING_MODEL = "models/gemini-embedding-001"

# RAGに使用する埋め込みモデル設定
RAG_EMBEDDING_MODEL = "models/gemini-embedding-001"

class GeminiRAGEvaluator(ABC):
    """Geminiを用いたRAGの評価用基底クラス"""

    def __init__(self, model_name: str):
        """
        Args:
            model_name: RAGに使用するGeminiモデルの名前
        """
        self.model_name = model_name
        self._api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "GOOGLE_API_KEY または GEMINI_API_KEY が設定されていません。.envファイルを確認してください。"
            )

        self._llm = None
        self._vector_store = None
        self._retriever = None

    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """LLMを初期化する"""
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self._api_key,
        )

    def _initialize_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """埋め込みモデルを初期化する"""
        return GoogleGenerativeAIEmbeddings(
            model=RAG_EMBEDDING_MODEL,
            google_api_key=self._api_key,
        )
    
    def _initialize_evaluator_llm(self) -> InstructorBaseRagasLLM:
        """評価用LLMを初期化する"""
        client = genai.Client(api_key=self._api_key)
        return llm_factory(
            model=EVALUATOR_MODEL, 
            provider="google", 
            client=client
        )

    def _initialize_evaluator_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """評価用埋め込みモデルを初期化する"""
        return GoogleGenerativeAIEmbeddings(
            model=EVALUATOR_EMBEDDING_MODEL,
            google_api_key=self._api_key,
        )

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

        pattern = re.compile(r"^gemini-3-.+$")
        if bool(pattern.match(self.model_name)):
            # Gemini 3系の場合、contentが[{'type': 'text', 'text': '...', 'extras': {'signature': ...}}]の形式になる
            return response.content[0]['text'], execution_time
        else:
            # Gemini 2系の場合、contentが文字列になる
            return response.content, execution_time

    def _query_rag(self, question: str, prompt: ChatPromptTemplate) -> tuple[str, str, float]:
        """RAGで質問に回答し、実行時間を計測する"""
        start_time = time.time()

        # コンテキストを検索
        docs = self._retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # LLMで回答を生成
        messages = prompt.format_messages(context=context, question=question)
        response = self._llm.invoke(messages)

        execution_time = time.time() - start_time

        pattern = re.compile(r"^gemini-3-.+$")
        if bool(pattern.match(self.model_name)):
            # Gemini 3系の場合、contentが[{'type': 'text', 'text': '...', 'extras': {'signature': ...}}]の形式になる
            return response.content[0]['text'], context, execution_time
        else:
            # Gemini 2系の場合、contentが文字列になる
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
        evaluation_result = evaluate(dataset=dataset, metrics=metrics, embeddings=embeddings)
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
        evaluation_result = evaluate(dataset=dataset, metrics=[metric], embeddings=embeddings)

        return evaluation_result.to_pandas()["answer_relevancy"].tolist()

    def _save_results(self, results: list[dict], ragas_scores: pd.DataFrame) -> None:
        """評価結果を保存する"""
        # results_{model_name}.xlsxの作成
        results_df = pd.DataFrame(results)

        # Ragasスコアを追加
        results_df["rag.context_precision"] = ragas_scores["context_precision"]
        results_df["rag.context_recall"] = ragas_scores["context_recall"]
        results_df["rag.context_entities_recall"] = ragas_scores["context_entity_recall"]
        results_df["rag.noise_sensitivity"] = ragas_scores["noise_sensitivity(mode=relevant)"]
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

    def _update_summary(self, results_df: pd.DataFrame) -> None:
        """summary.xlsxを更新する"""
        # 平均値を計算
        summary_row = {
            "model": self.model_name,
            "llm.execution_time": results_df["llm.execution_time"].mean(),
            "rag.execution_time": results_df["rag.execution_time"].mean(),
            "llm.response_relevancy": results_df["llm.response_relevancy"].mean(),
            "rag.context_precision": results_df["rag.context_precision"].mean(),
            "rag.context_recall": results_df["rag.context_recall"].mean(),
            "rag.context_entities_recall": results_df["rag.context_entities_recall"].mean(),
            "rag.noise_sensitivity": results_df["rag.noise_sensitivity"].mean(),
            "rag.response_relevancy": results_df["rag.response_relevancy"].mean(),
            "rag.faithfulness": results_df["rag.faithfulness"].mean(),
        }

        # 既存のsummary.xlsxを読み込む or 新規作成
        if SUMMARY_OUTPUT_PATH.exists():
            summary_df = pd.read_excel(SUMMARY_OUTPUT_PATH, engine="openpyxl")
            # 同じモデルの行が存在する場合は削除
            summary_df = summary_df[summary_df["model"] != self.model_name]
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
        print(f"=== Gemini RAG評価: {self.model_name} ===")

        # RAGチェーンの構築
        print("\n1. RAGチェーンの構築")
        prompt = self._build_rag_chain()
        print(f"  モデル: {self.model_name}")
        print(f"  埋め込みモデル: {RAG_EMBEDDING_MODEL}")

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
