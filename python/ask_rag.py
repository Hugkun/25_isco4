"""
ask_rag.py - RAGを直接実行するスクリプト

RagExecutorクラスを定義し、RAGチェーンを構築して質問に回答する。
"""

import argparse
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

# .envファイルの読み込み
load_dotenv()

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = PROJECT_ROOT / "outputs" / "faiss.index"

# デフォルト設定
DEFAULT_RAG_EMBEDDING_MODEL = "embeddinggemma:latest"

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

class RagExecutor:
    """RAG実行クラス"""

    def __init__(
        self,
        model_name: str,
        rag_embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
    ):
        """
        Args:
            model_name: RAGに使用するLLMの名前
            rag_embedding_model: RAG用埋め込みモデルの名前（デフォルト: embeddinggemma:latest）
        """
        self.model_name = model_name
        self.rag_embedding_model = rag_embedding_model

        # Gemini API用のキー（LLMや埋め込みモデルで使用する可能性がある）
        self._api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        self._llm = None
        self._vector_store = None
        self._retriever = None

    def _initialize_llm(self) -> ChatGoogleGenerativeAI | ChatOllama:
        """LLMを初期化する"""
        if self.model_name in GEMINI_LLM_MODELS:
            # Gemini APIを使用
            if not self._api_key:
                raise ValueError(
                    "GOOGLE_API_KEY または GEMINI_API_KEY が設定されていません。.envファイルを確認してください。"
                )
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self._api_key,
            )
        else:
            # Ollamaを使用
            return ChatOllama(model=self.model_name)

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

    def _initialize_embeddings(
        self,
    ) -> GoogleGenerativeAIEmbeddings | OllamaEmbeddings:
        """埋め込みモデルを初期化する"""
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

        # レスポンスの形式に応じて回答を抽出
        pattern = re.compile(r"^gemini-3-.+$")
        if bool(pattern.match(self.model_name)):
            # Gemini 3系の場合、contentが[{'type': 'text', 'text': '...', 'extras': {'signature': ...}}]の形式になる
            return response.content[0]["text"], context, execution_time
        else:
            # Gemini 2系またはOllamaの場合、contentが文字列になる
            return response.content, context, execution_time
        
    def execute_rag(self, question: str) -> None:
        """RAGを実行する

        Args:
            question: 質問文字列
        """
        print(f"=== RAG実行: {self.model_name} ===")
        print(f"埋め込みモデル: {self.rag_embedding_model}")

        prompt = self._build_rag_chain()

        answer, context, exec_time = self._query_rag(question, prompt)

        print("\n=== 質問 ===")
        print(question)
        print("\n=== 回答 ===")
        print(answer)
        print("\n=== コンテキスト ===")
        print(context)
        print("\n=== 実行時間 ===")
        print(f"{exec_time:.2f} 秒")


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description="RAGを直接実行する")
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        help="RAG用LLMの名前（例: gemini-2.5-flash, gemma3:12b）",
    )
    parser.add_argument(
        "--rag-embedding-model",
        type=str,
        default=DEFAULT_RAG_EMBEDDING_MODEL,
        help=f"RAG用埋め込みモデルの名前（デフォルト: {DEFAULT_RAG_EMBEDDING_MODEL}）",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="質問文（スペースを含む場合は引用符で囲んでください）",
    )
    return parser.parse_args()


def main():
    """メイン関数"""
    args = parse_args()

    rag = RagExecutor(
        model_name=args.llm,
        rag_embedding_model=args.rag_embedding_model,
    )
    rag.execute_rag(args.question)


if __name__ == "__main__":
    main()