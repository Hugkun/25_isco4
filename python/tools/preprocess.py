"""
preprocess.py - PDFファイルの前処理スクリプト

data/documents以下のPDFファイルを読み込み、
不要な改行と空白の削除、チャンク化を行い、
outputs/preprocessed/preprocessed.xlsxに保存する。
"""

import re
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "preprocessed"
OUTPUT_FILE = OUTPUT_DIR / "preprocessed.xlsx"

# チャンク設定
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def clean_text(text: str) -> str:
    """不要な改行と空白を削除する"""
    # 連続する空白を1つに
    text = re.sub(r"[ \t]+", " ", text)
    # 連続する改行を1つに
    text = re.sub(r"\n+", "\n", text)
    # 改行前後の空白を削除
    text = re.sub(r" ?\n ?", "\n", text)
    # 先頭・末尾の空白を削除
    text = text.strip()
    return text


def load_pdf_files() -> list:
    """data/documents以下の全PDFファイルを読み込む"""
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"警告: {INPUT_DIR} にPDFファイルが見つかりません")
        return []

    print(f"{len(pdf_files)}件のPDFファイルを検出しました")

    all_documents = []
    for pdf_path in sorted(pdf_files):
        print(f"  読み込み中: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        all_documents.extend(documents)

    return all_documents


def preprocess_documents(documents: list) -> list:
    """ドキュメントの前処理（テキストクリーニング）を行う"""
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    return documents


def chunk_documents(documents: list) -> list:
    """ドキュメントをチャンク化する"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def save_to_excel(chunks: list) -> None:
    """チャンクをExcelファイルに保存する"""
    # 出力ディレクトリの作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # データ仕様書に基づくカラム
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

    # DataFrameの作成
    records = []
    for chunk in chunks:
        record = {"page_content": chunk.page_content}
        for col in metadata_columns:
            record[col] = chunk.metadata.get(col, "")
        records.append(record)

    df = pd.DataFrame(records)

    # カラム順序を仕様書に合わせる
    columns_order = ["page_content"] + metadata_columns
    df = df[columns_order]

    # Excelファイルに保存
    df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
    print(f"保存完了: {OUTPUT_FILE}")
    print(f"  チャンク数: {len(chunks)}")


def main():
    """メイン処理"""
    print("=== PDF前処理スクリプト ===")

    # PDFファイルの読み込み
    print("\n1. PDFファイルの読み込み")
    documents = load_pdf_files()
    if not documents:
        print("処理を終了します")
        return

    # テキストクリーニング
    print("\n2. テキストクリーニング")
    documents = preprocess_documents(documents)
    print(f"  ドキュメント数: {len(documents)}")

    # チャンク化
    print("\n3. チャンク化")
    chunks = chunk_documents(documents)
    print(f"  チャンク数: {len(chunks)}")

    # Excelファイルに保存
    print("\n4. Excelファイルに保存")
    save_to_excel(chunks)

    print("\n=== 処理完了 ===")


if __name__ == "__main__":
    main()
