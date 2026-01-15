from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import time
import os
import re
import argparse

# .envファイルの読み込み
load_dotenv()

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = PROJECT_ROOT / "outputs" / "faiss.index"

# RAGに使用する埋め込みモデル設定
RAG_EMBEDDING_MODEL = "models/gemini-embedding-001"

class rag_executor:
    """RAG実行クラス"""

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

    def _initialize_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """埋め込みモデルを初期化する"""
        return GoogleGenerativeAIEmbeddings(
            model=RAG_EMBEDDING_MODEL,
            google_api_key=self._api_key,
        )
    
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
        
    def execute_rag(self, question: list):
        """RAGを実行する"""
        prompt = self._build_rag_chain()

        answer, context, exec_time = self._query_rag(question, prompt)

        print("\n=== 質問 ===")
        print(question)
        print("\n=== 回答 ===")
        print(answer)
        print("\n=== コンテキスト ===")
        print(context)
        print("\n=== 実行時間 ===")
        print(f"{exec_time} 秒")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="RAG実行スクリプト"
    )

    parser.add_argument(
        "model_name",
        help="モデル名 (e.g. gemini-3-pro, gemini-3-flash)"
    )

    parser.add_argument(
        "question",
        help="質問文 (スペースを含む場合は引用符で囲んでください)"
    )

    args = parser.parse_args()

    model_name = args.model_name
    question = args.question

    rag = rag_executor(model_name)
    rag.execute_rag(question)

    
if __name__ == "__main__":
    main()