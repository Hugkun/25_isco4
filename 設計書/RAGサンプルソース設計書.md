# RAGサンプルソース設計書

## プロジェクトの目的
このプロジェクトの目的は、RAGのサンプルソースを作成することです。複数の文章生成モデルを使ったRAGの性能をまとめて評価できるシステムを作ります。

## フォルダ構成
```
25_isco4_dev
├── CLAUDE.md                                                                 # claude codeに対するシステムプロンプト
├── README.md                                                                 # 開発環境について簡単に記載
├── data                                                                      # データの管理
│   ├── documents                                                             # RAGのデータソース
│   │   ├── 01_就業規則.pdf
│   │   └── ...
│   └── valid.xlsx                                                            # RAGの検証用データ
├── notebooks                                                                 # Jupyter Notebookの保存
│   └── visualize.ipynb                                                       # グラフの表示（人手で作成）
├── outputs                                                                   # スクリプトの出力の保存
│   ├── external
│   │   └── gemini
│   │       ├── results_gemini_3_pro.xslx                                     # Gemini 3 Proを用いたRAGの評価結果
│   │       ├── results_gemini_3_flash.xslx                                   # Gemini 3 Flashを用いたRAGの評価結果
│   │       ├── results_gemini_2.5_flash.xslx                                 # Gemini 2.5 Flashを用いたRAGの評価結果
│   │       ├── results_gemini_2.5_flash_lite.xslx                            # Gemini 2.5 Flash Liteを用いたRAGの評価結果
│   │       └── results_gemini_2.5_pro.xslx                                   # Gemini 2.5 proを用いたRAGの評価結果
│   ├── faiss.index                                                           # Faissのベクトルデータベースの情報
│   ├── plots                                                                 # visualize.pyの出力の保存
│   │   ├── execution_time_comparison.png                                     # 平均実行時間の棒グラフ
│   │   ├── noise_sensitivity_model_comparison.png                            # 平均Noise Sensitivityの棒グラフ
│   │   ├── response_relevancy_comparison.png                                 # 平均Response Relevancyの棒グラフ
│   │   ├── faithfulness_comparison.png                                       # 平均Faithfulnessの棒グラフ
│   │   ├── execution_time_boxplot.png                                        # 実行時間の箱ひげ図
│   │   ├── noise_sensitivity_model_boxplot.png                               # Noise Sensitivityの箱ひげ図
│   │   ├── response_relevancy_boxplot.png                                    # Response Relevancyの箱ひげ図
│   │   └── faithfulness_boxplot.png                                          # Faithfulnessの箱ひげ図
│   ├── preprocessed
│   │   └── preprocessed.xlsx                                                 # 前処理済みデータ
│   └── summary.xlsx                                                          # RAG評価結果のまとめ
├── pyproject.toml                                                            # uv環境定義
├── python                                                                    # Pythonスクリプトの保存
│   ├── ask_rag.py                                                            # RAGを1回だけ実行するスクリプト
│   ├── external
│   │   └── gemini                                                            # Geminiを用いたRAGの評価用スクリプトの保存
│   │       ├── evaluate_gemini_base.py                                       # Geminiを用いたRAGの評価用スクリプトの共通クラスの定義
│   │       ├── evaluate_gemini_3_pro.py                                      # Gemini 3 Proを用いたRAGの評価用スクリプト
│   │       ├── evaluate_gemini_3_flash.py                                    # Gemini 3 Flashを用いたRAGの評価用スクリプト
│   │       ├── evaluate_gemini_2.5_flash.py                                  # Gemini 2.5 Flashを用いたRAGの評価用スクリプト
│   │       ├── evaluate_gemini_2.5_flash_lite.py                             # Gemini 2.5 Flash Liteを用いたRAGの評価用スクリプト
│   │       └── evaluate_gemini_2.5_pro.py                                    # Gemini 2.5 proを用いたRAGの評価用スクリプト
│   ├── tools
│   │   ├── ingest.py                                                         # チャンクのベクトル化・ベクトルデータベースの作成と保存
│   │   └── preprocess.py                                                     # 文書の前処理（データクリーニング・チャンキング）
│   └── visualize
│       └── visualize.py                                                      # グラフの作成
├── shell                                                                     # Shellスクリプトの保存
│   ├── preprocess_ingest.sh                                                  # データの前処理、ベクトルデータベースの作成を実行
│   └── evaluate.sh                                                           # RAGの構築と評価を実行
├── uv.lock                                                                   # uv環境定義
├── 設計書                                                                     # 設計書の保存
│   ├── RAGサンプルソース設計書.md
│   ├── グラフ仕様書.md
│   ├── データ仕様書.md
│   └── スクリプト仕様書.md
└── 作業メモ       
    ├── 25_0108_hayate.md                                                     # 作業メモの保存
    └── ...
```

## システムの全体フロー
### RAGの性能評価
1. ユーザーがpreprocess_ingest.shを実行する。
2. preprocess_ingest.shがpreprocess.pyを実行する。
3. preprocess.pyがdata/documentsフォルダ内の全てのpdfファイルを読み取り、データ前処理を実行する。
4. preprocess.pyが前処理を実行した結果をoutputs/preprocessed/preprocessed.xlsxに保存する
（ここで、preprocess.pyの処理は終了）。
5. preprocess_ingest.shがingest.pyを実行する。
6. ingest.pyがoutputs/preprocessed/preprocessed.xlsxを読み込み、埋め込みモデルを使ってチャンク化された全ての文章をベクトルに変換する。
7. ingest.pyが全ての文章ベクトルをベクトルデータベースに保存する。
8. ingest.pyがベクトルデータベースの情報をoutputs/index.faissに保存する（ここで、ingest.pyの処理は終了）。
9. ユーザーがevaluate.pyを実行する。
9. evaluate.shがevaluate_{model_name}.pyを実行する。
    1. ユーザーがevaluate.shの引数でRAGで使用するモデル名を指定していた場合は、そのモデルに対応するevaluate_{model_name}.pyのみ実行する。
    2. ユーザーがRAGに使用するモデルを何も指定していなかった場合は、全てのevaluate_{model_name}.pyを実行する。
10. 各evaluate_{model_name}.pyがdata/validation.xlsxを読み込みRagasを用いてRAGなしのモデルの評価を実行する。
11. 各evaluate_{model_name}.pyがoutputs/index.faissを読み込み、RAGを構築する。
12. 各evaluate_{model_name}.pyがRagasを用いてRAGの評価を実行する。
13. 各evaluate_{model_name}.pyが、評価の全ログをoutputs/external/gemini/results_{model_name}.xlsxに保存し、評価の要約をoutputs/summary.xlsxに追記する（ここで、evaluate_{model_name}.pyの処理は終了）。

### RAGの実行
1. ユーザーがpreprocess_ingest.shを実行する。
2. ユーザーがモデル名と質問文を引数に渡して、ask_rag.pyを実行する。
3. ask_rag.pyが、outputs/index.faissを読み込み、引数のモデル名に基づいてRAGを構築する。
4. ask_rag.pyが、引数の質問文に基づいてRAGから回答を生成し、標準出力にprintする。

### 評価の可視化
1. ユーザーがpreprocess_ingest.shを実行する。
2. ユーザーがvisualize.ipynbを実行する。
3. visualize.ipynbがvisualize.pyからVisualizationGeneratorクラスを読み込みインスタンス化する。
4. visualize.ipynbがVisualizationGeneratorクラスのgenerate_all_visualizationsメソッドを呼び出す。
5. generate_all_visualizationsメソッドが、outputs/summary.xlsxを読み込み、グラフを描画してoutputs/figures内に保存する。
6. visualize.ipynbがoutputs/figures内に保存されたグラフを読み込みノートブック内で表示する。

## スクリプト仕様
リンク先を参照してください。
* [スクリプト仕様書](./スクリプト仕様書.md)

## データ仕様
リンク先を参照してください。
* [データ仕様書](./データ仕様書.md)

## グラフ仕様
リンク先を参照してください。
* [グラフ仕様書](./グラフ仕様書.md)

## 評価方法
### 使用するRAG評価ツール
Ragas

### 使用する評価指標
* Context Precision
* Context Recall
* Context Entities Recall
* Noise Sensitivity
* Response Relevancy
* Faithfulness

### 評価用LLM
Gemini 2.5 Flash-Lite

### 評価用埋め込みモデル
gemini-embedding-001

## 使用するAPI
Google AI Studio Gemini API

APIキーは.envで管理する。

## 使用する文章生成モデル
* Gemini 3 Pro
* Gemini 3 Flash
* Gemini 2.5 Flash
* Gemini 2.5 Flash-Lite
* Gemini 2.5 Pro

## 使用する埋め込みモデル
gemini-embedding-001

## 使用するベクトルデータベース
faiss

## 動作環境
OSはMac / Windows / Linuxで動くことを想定しています。

## 環境セットアップ
### 必要条件
* Python 3.14.2
* uv (パッケージマネージャー)

### 開発者向け環境（Windows/macOS/Linux）
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

# Git
## 作業ブランチ
```
feature/add_2026_0116_ver_hayate
```

このブランチ以外に変更を加えないでください。


## コミット
都度適切なタイミングでコミットしてください。

## 備考
visualize.ipynbは人間が作成します。