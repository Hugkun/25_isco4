### 生成AIを活用した解決プロセス体験ワークショップ 
# 第４回 課題解決に向けたベースライン開発

## このリポジトリの説明
RAGを構築し、評価し、動かすまでの機能を備えたシステムのソースコードです。受講生の皆様のシステム開発の参考としてお使いいただけます。  
このソースコードを起点に、開発を始めていただいても問題ありません。  

## このシステムの使い方
### 1. データの前処理とベクトルデータベースの作成
シェルスクリプトを実行することで、自動的に実行されます。

シェルスクリプトの実行:
```
cd shell
sh preprocess_ingest.sh
```

### 2. RAGシステムの構築と評価
シェルスクリプトを実行することで、自動的に実行されます。

すべての利用可能なLLMを使ってRAGを構築し、評価する場合:
```
cd shell
sh evaluate.sh
```

特定のLLMを使用したRAGを構築して評価する場合:
```
cd shell
sh evaluate.sh <model_name>
```

\<model-name\>には現在次の項目を指定できます。

**対応モデル**:
- gemini-2.5-flash-lite
- gemini-2.5-flash
- gemini-2.5-pro
- gemini-3-flash-preview
- gemini-3-pro-preview

使用したLLMごとの評価ログは、`outputs/external/gemini/`に出力されます。

評価のまとめは、`outputs/summary.xlsx`に出力されます。

### 3. 評価結果の可視化
Jupyter Labを実行し、`notebooks/visualize.ipynb`を開きます。

Jupyter Labの実行:
```
uv run jupyter-lab
```

ノートブックのセルを順番に実行することで、評価結果の可視化を見ることができます。

あるいは、別の方法として、Pythonスクリプトを直接実行することでも、評価結果を可視化することができます。

可視化用Pythonスクリプトの実行:
```
cd python/visualize
uv run python visualize.py
```

可視化結果は、`outputs/plots/`に画像ファイルとして出力されます。

### 4. RAGの実行
RAGを直接実行して、質問を投げかけることもできます。　　
先に、データの前処理とベクトルデータベースの作成を済ませておく必要があります。

RAGの実行:
```
cd python
uv run python ask_rag.py <model_name> <question>
```

\<model-name\>には、「2. RAGシステムの構築と評価」で使用できるモデルと、同じモデルを使用できます。

\<question\>には自由な質問を含めることができます。

## データの置き場所
### 元データ
RAGに参照させるデータは、`data/documents`以下に配置します。読み取れるデータは**pdfファイル**です。

### 前処理済みデータ
前処理を施した後のデータは、`outputs/preprocessed/preprocessed.xlsx`に保存されます。

このファイルを直接編集することで、RAGに与える参照文書の内容を変化させることが可能です。

編集した内容を反映させたい場合は、エクセルを上書き保存したのち、Pythonスクリプトを実行します。

Pythonスクリプトの実行:
```
cd python/tools
uv run python ingest.py
```

## 環境構築の方法
### 必要条件
* Python 3.14.2
* uv (パッケージマネージャー)

### 手順
1. uvのインストール
   ```bash
   # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. プロジェクトセットアップ
   ```bash
   uv sync
   ```

3. APIキーの登録
   ```bash
   cp .env.sample .env
   ```
   `GEMINI_API_KEY`に自身のAPIキーを登録。