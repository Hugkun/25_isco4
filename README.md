### 生成AIを活用した解決プロセス体験ワークショップ 
# 第４回 課題解決に向けたベースライン開発

## このリポジトリの説明
RAGを構築し、評価し、動かすまでの機能を備えたシステムのソースコードです。受講生の皆様のシステム開発の参考としてお使いいただけます。  
このソースコードを起点に、開発を始めていただいても問題ありません。  

## このシステムの使い方

### 1. データの前処理とベクトルデータベースの作成
シェルスクリプトを実行することで、自動的に実行されます。

標準の埋め込みモデルを使用する場合:
```bash
cd shell
bash preprocess_ingest.sh
```

埋め込みモデルを指定する場合:
```bash
cd shell
bash preprocess_ingest.sh --embedding-model <model-name>
```

### 2. RAGシステムの構築と評価
シェルスクリプトを実行することで、自動的に実行されます。

標準のRAG用LLM・RAG用埋め込みモデル・評価用LLM・評価用埋め込みモデルを使用する場合:
```bash
cd shell
bash evaluate.sh
```

RAG用LLM・RAG用埋め込みモデル・評価用LLM・評価用埋め込みモデルを指定する場合:
```bash
cd shell
bash evaluate.sh \
   --llm <model-name> \
   --rag-embedding-model <model-name> \
   --evaluator-llm <model-name> \
   --evaluator-embedding-model <model-name>
```

使用したLLMごとの評価ログは、`<project_root>/outputs/external/gemini/`または`outputs/external/ollama`に出力されます。

評価のまとめは、`<project_root>/outputs/summary.xlsx`に出力されます。

### 3. 評価結果の可視化
Jupyter Labを実行し、`<project_root>/notebooks/visualize.ipynb`を開きます。

Jupyter Labの実行:
```bash
uv run jupyter-lab
```

ノートブックのセルを順番に実行することで、評価結果の可視化を見ることができます。

あるいは、別の方法として、Pythonスクリプトを直接実行することでも、評価結果を可視化することができます。

可視化用Pythonスクリプトの実行:
```bash
cd python/visualize
uv run python visualize.py
```

可視化結果は、`<project_root>/outputs/plots/`に画像ファイルとして出力されます。

### 4. RAGの実行
RAGを直接実行して、質問を投げかけることもできます。　　
先に、データの前処理とベクトルデータベースの作成を済ませておく必要があります。

RAGの実行:
```bash
cd python
uv run python ask_rag.py --llm <model_name> --rag-embedding-model <model_name> --question <question>
```

\<question\>には自由な質問を含めることができます。空白を含む場合は引用符で囲みます。

## データの置き場所

### 元データ
RAGに参照させるデータは、`<project_root>/data/documents`以下に配置します。読み取れるデータは**pdfファイル**です。

### 前処理済みデータ
前処理を施した後のデータは、`<project_root>/outputs/preprocessed/preprocessed.xlsx`に保存されます。

このファイルを直接編集することで、RAGに与える参照文書の内容を変化させることが可能です。

編集した内容を反映させたい場合は、エクセルを上書き保存したのち、Pythonスクリプトを実行します。

Pythonスクリプトの実行:
```bash
cd python/tools
uv run python ingest.py
```

Pythonスクリプトの実行(埋め込みモデル指定あり):
```bash
cd python/tools
uv run python ingest.py --embedding-model <model-name>
```

## 対応モデル

### RAG用LLM

- gemini-2.5-flash-lite
- gemini-2.5-flash
- gemini-2.5-pro
- gemini-3-flash-preview
- gemini-3-pro-preview
- gemma3:1b
- gemma3:4b
- gemma3:12b
- gemma3:27b
- gpt-oss:20b
- gpt-oss:120b


### RAG用埋め込みモデル

- gemini-embedding-001
- Ollama上で公開されている埋め込みモデル

### 評価用LLM

- gemini-2.5-flash-lite
- gemini-2.5-flash
- gemini-2.5-pro
- gemini-3-flash-preview
- gemini-3-pro-preview
- Ollama上で公開されているLLM

### 評価用埋め込みモデル

- gemini-embedding-001
- Ollama上で公開されている埋め込みモデル

## 環境構築の方法
### 必要条件
- Python 3.13
- uv (パッケージマネージャー)
- Ollama

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

4. Ollamaのインストール
   [公式ページ](https://ollama.com/download)からインストーラをダウンロードしてインストール。

5. ローカルモデルのpull
   必要なローカルモデルをpullする。
   
   ```bash
   ollama pull <model-name>
   ```

## その他
### タイムアウトエラー対処法
評価の実行中にタイムアウトエラーが出る場合は、Ragasのタイムアウト時間を変更して、再度実行し直してみてください。
Ragasでの評価が正常に実行されないと、該当のスコアはNanとしてエクセルに記録されるため注意してください。

#### GeminiモデルのLLM単体の回答の評価でタイムアウトする場合
`<project_root>/python/external/gemini/evaluate_gemini_base.py`の`_evaluate_llm_response_relevancy`関数内のtimeout設定を変更（デフォルト3000秒）:

```
evaluation_result = evaluate(
   dataset=dataset, metrics=[metric], embeddings=embeddings, run_config=RunConfig(timeout=3000.0)
) # タイムアウト3000秒
```

#### GeminiモデルのRAGの回答の評価でタイムアウトする場合
`<project_root>/python/external/gemini/evaluate_gemini_base.py`の`_evaluate_with_ragas`関数内のtimeout設定を変更（デフォルト3000秒）:

```
evaluation_result = evaluate(
   dataset=dataset, metrics=metrics, embeddings=embeddings, run_config=RunConfig(timeout=3000.0)
) # タイムアウト3000秒
```

#### OllamaモデルのLLM単体の回答の評価でタイムアウトする場合
`<project_root>/python/external/ollama/evaluate_ollama_base.py`の`_evaluate_llm_response_relevancy`関数内のtimeout設定を変更（デフォルト3000秒）:

```
evaluation_result = evaluate(
   dataset=dataset, metrics=[metric], embeddings=embeddings, run_config=RunConfig(timeout=3000.0)
) # タイムアウト3000秒
```

#### OllamaモデルのRAGの回答の評価でタイムアウトする場合
`<project_root>/python/external/ollama/evaluate_ollama_base.py`の`_evaluate_with_ragas`関数内のtimeout設定を変更（デフォルト3000秒）:

```
evaluation_result = evaluate(
   dataset=dataset, metrics=metrics, embeddings=embeddings, run_config=RunConfig(timeout=3000.0)
) # タイムアウト3000秒
```


### OMPエラー対処法
次のようなエラーが出る場合は、.envに`KMP_DUPLICATE_LIB_OK=True`を追加してください。

```
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://openmp.llvm.org/
```