# RAGサンプルソース設計書

ver1のRAGサンプルソースの設計書は、[こちら](../ver1/RAGサンプルソース設計書.md)を参照してください。

## プロジェクトの目的
このプロジェクトの目的は、RAGのサンプルソースを作成することです。複数のLLMを使ったRAGの性能をまとめて評価できるシステムを作ります。

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
│   │   ├── gemini
│   │   │   ├── results_gemini-3-pro.xslx                                     # Gemini 3 Proを用いたRAGの評価結果
│   │   │   ├── results_gemini-3-flash.xslx                                   # Gemini 3 Flashを用いたRAGの評価結果
│   │   │   ├── results_gemini-2.5-flash.xslx                                 # Gemini 2.5 Flashを用いたRAGの評価結果
│   │   │   ├── results_gemini-2.5-flash-lite.xslx                            # Gemini 2.5 Flash Liteを用いたRAGの評価結果
│   │   │   └── results_gemini-2.5-pro.xslx                                   # Gemini 2.5 proを用いたRAGの評価結果
│   │   └── ollama
│   │       ├── results_gemma3:1b.xslx                                        # gemma3:1bを用いたRAGの評価結果
│   │       ├── results_gemma3:4b.xslx                                        # gemma3:4bを用いたRAGの評価結果
│   │       ├── results_gemma3:12b.xslx                                       # gemma3:12bを用いたRAGの評価結果
│   │       ├── results_gemma3:27b.xslx                                       # gemma3:27bを用いたRAGの評価結果
│   │       ├── results_gpt-oss:20b.xslx                                      # gpt-oss:20bを用いたRAGの評価結果
│   │       └── results_gpt-oss:120b.xslx                                     # gpt-oss:120bを用いたRAGの評価結果
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
│   └── summary.xlsx                                                          # 全てのRAGの評価結果のまとめ
├── pyproject.toml                                                            # uv環境定義
├── python                                                                    # Pythonスクリプトの保存
│   ├── ask_rag.py                                                            # RAGを1回だけ実行するスクリプト
│   ├── external
│   │   ├── gemini                                                            # Geminiを用いたRAGの評価用スクリプトの保存
│   │   │   ├── evaluate_gemini_base.py                                       # Geminiを用いたRAGの評価用スクリプトの共通クラスの定義
│   │   │   ├── evaluate_gemini_3_pro.py                                      # Gemini 3 Proを用いたRAGの評価用スクリプト
│   │   │   ├── evaluate_gemini_3_flash.py                                    # Gemini 3 Flashを用いたRAGの評価用スクリプト
│   │   │   ├── evaluate_gemini_2.5_flash.py                                  # Gemini 2.5 Flashを用いたRAGの評価用スクリプト
│   │   │   ├── evaluate_gemini_2.5_flash_lite.py                             # Gemini 2.5 Flash Liteを用いたRAGの評価用スクリプト
│   │   │   └── evaluate_gemini_2.5_pro.py                                    # Gemini 2.5 proを用いたRAGの評価用スクリプト
│   │   └── ollama                                                            # Ollamaに公開されているモデルを用いたRAGの評価用スクリプトの保存
│   │       ├── evaluate_ollama_base.py                                       # Ollamaのモデルを用いたRAGの評価用スクリプトの共通クラスの定義
│   │       ├── evaluate_gemma3_1b.py                                         # gemma3:1bを用いたRAGの評価用スクリプト
│   │       ├── evaluate_gemma3_4b.py                                         # gemma3:4bを用いたRAGの評価用スクリプト
│   │       ├── evaluate_gemma3_12b.py                                        # gemma3:12bを用いたRAGの評価用スクリプト
│   │       ├── evaluate_gemma3_27b.py                                        # gemma3:27bを用いたRAGの評価用スクリプト
│   │       ├── evaluate_gpt-oss_20b.py                                       # gpt-oss:20bを用いたRAGの評価用スクリプト
│   │       └── evaluate_gpt-oss_120b.py                                      # gpt-oss:120bを用いたRAGの評価用スクリプト
│   ├── tools
│   │   ├── ingest.py                                                         # チャンクのベクトル化・ベクトルデータベースの作成と保存
│   │   └── preprocess.py                                                     # 文書の前処理（データクリーニング・チャンキング）
│   ├── utils
│   │   └── hardware_info.py                                                  # ハードウェア情報の取得
│   └── visualize
│       └── visualize.py                                                      # グラフの作成
├── shell                                                                     # Shellスクリプトの保存
│   ├── preprocess_ingest.sh                                                  # データの前処理、ベクトルデータベースの作成を実行
│   └── evaluate.sh                                                           # RAGの構築と評価を実行
├── uv.lock                                                                   # uv環境定義
├── 設計書                                                                     # 設計書の保存
│   ├── ver1
│   │   ├── RAGサンプルソース設計書.md
│   │   ├── グラフ仕様書.md
│   │   ├── データ仕様書.md
│   │   └── スクリプト仕様書.md
│   └── ver2
│       ├── RAGサンプルソース設計書ver2.md
│       ├── クラス仕様書ver2.md
│       ├── グラフ仕様書ver2.md
│       ├── データ仕様書ver2.md
│       └── スクリプト仕様書ver2.md
└── 作業メモ       
    ├── 26_0108_hayate.md                                                     # 作業メモの保存
    └── ...
```

## スクリプト仕様
以下を参照してください。

[スクリプト仕様書](./スクリプト仕様書ver2.md)

## データ仕様
以下を参照してください。

[データ仕様書](./データ仕様書ver2.md)

## クラス仕様書
以下を参照してください。

[クラス仕様書](./クラス仕様書ver2.md)

## グラフ仕様書
以下を参照してください。

[グラフ仕様書](./グラフ仕様書ver2.md)

## API
### 使用するAPI名
Google AI Studio Gemini API

### APIキー管理方法
Gemini API用のAPIキーは`<project_root>/.env`に`GEMINI_API_KEY`という名前で保存する。

## ツール

### LLMのローカル実行ツール
Ollama

### LLM評価ツール
Ragas

### ベクトルデータベース
Faiss

## 環境セットアップ
### 必要条件
- Python 3.13
- uv (パッケージマネージャー)

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

## Git
### 作業ブランチ
```
feature/add_ollama_hayate
```

このブランチ以外に変更を加えないでください。