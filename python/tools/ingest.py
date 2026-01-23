"""
ingest.py - チャンクのベクトル化・ベクトルデータベース作成スクリプト

outputs/preprocessed/preprocessed.xlsxを読み込み、
埋め込みモデルを使って文章をベクトル化し、
FAISSベクトルデータベースを作成してoutputs/faiss.indexに保存する。
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# .envファイルの読み込み
load_dotenv()

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "outputs" / "preprocessed" / "preprocessed.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "faiss.index"

# デフォルト埋め込みモデル設定
DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"

# Gemini APIの埋め込みモデル名
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


def load_preprocessed_data() -> list[Document]:
    """前処理済みデータを読み込み、Documentオブジェクトのリストに変換する"""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    print(f"  読み込み完了: {len(df)}件のチャンク")

    # メタデータカラム
    metadata_columns = [
        "producer",
        "creator",
        "creationdate",
        "author",
        "keywords",
        "moddate",
        "subject",
        "title",
        "trapped",
        "source",
        "total_pages",
        "page",
        "page_label",
    ]

    # DocumentオブジェクトのリストSnに変換
    documents = []
    for _, row in df.iterrows():
        metadata = {}
        for col in metadata_columns:
            if col in row and pd.notna(row[col]):
                metadata[col] = row[col]

        doc = Document(page_content=str(row["page_content"]), metadata=metadata)
        documents.append(doc)

    return documents


def create_embeddings(
    model_name: str,
) -> GoogleGenerativeAIEmbeddings | OllamaEmbeddings:
    """埋め込みモデルを初期化する

    Args:
        model_name: 埋め込みモデルの名前。Gemini APIのgemini-embedding-001か、
                   Ollamaに公開されているモデルを指定できる。

    Returns:
        初期化された埋め込みモデル
    """
    if model_name == GEMINI_EMBEDDING_MODEL:
        # Gemini APIの埋め込みモデルを使用
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY または GEMINI_API_KEY が設定されていません。.envファイルを確認してください。"
            )
        embeddings = GoogleGenerativeAIEmbeddings(
            model=f"models/{GEMINI_EMBEDDING_MODEL}", google_api_key=api_key
        )
    else:
        # Ollamaの埋め込みモデルを使用
        embeddings = OllamaEmbeddings(model=model_name)

    return embeddings


def create_vector_store(
    documents: list[Document], embeddings: GoogleGenerativeAIEmbeddings | OllamaEmbeddings
) -> FAISS:
    """FAISSベクトルストアを作成する"""
    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vector_store


def save_vector_store(vector_store: FAISS) -> None:
    """ベクトルストアをファイルに保存する"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(OUTPUT_FILE))
    print(f"  保存完了: {OUTPUT_FILE}")


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description="前処理済みデータを読み込み、ベクトルデータベースを作成する"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"埋め込みモデルの名前。Ollamaのモデルまたは'{GEMINI_EMBEDDING_MODEL}'を指定可能。"
        f"(デフォルト: {DEFAULT_EMBEDDING_MODEL})",
    )
    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_args()
    embedding_model = args.embedding_model

    print("=== ベクトルデータベース作成スクリプト ===")

    # 前処理済みデータの読み込み
    print("\n1. 前処理済みデータの読み込み")
    documents = load_preprocessed_data()

    # 埋め込みモデルの初期化
    print("\n2. 埋め込みモデルの初期化")
    embeddings = create_embeddings(embedding_model)
    print(f"  モデル: {embedding_model}")

    # ベクトルストアの作成
    print("\n3. ベクトルストアの作成")
    print("  ベクトル化中...")
    vector_store = create_vector_store(documents, embeddings)
    print(f"  ベクトル化完了: {len(documents)}件")

    # ベクトルストアの保存
    print("\n4. ベクトルストアの保存")
    save_vector_store(vector_store)

    print("\n=== 処理完了 ===")


if __name__ == "__main__":
    main()
