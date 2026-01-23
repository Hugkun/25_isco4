"""
visualize.py - RAG評価結果の可視化スクリプト

グラフ仕様書に基づいたグラフの画像を作成する。
VisualizationGeneratorクラスを定義し、generate_all_visualizationsメソッドで
outputs/summary.xlsxとresults_{model_name}.xlsxを読み込み、
全てのグラフをoutputs/plots以下に保存する。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SUMMARY_FILE = BASE_DIR / "outputs" / "summary.xlsx"
GEMINI_RESULTS_DIR = BASE_DIR / "outputs" / "external" / "gemini"
OLLAMA_RESULTS_DIR = BASE_DIR / "outputs" / "external" / "ollama"
OUTPUT_DIR = BASE_DIR / "outputs" / "plots"

# グラフの色設定
COLOR_NO_RAG = "#4A90D9"  # RAGなし（青系）
COLOR_RAG = "#E07B39"  # RAGあり（オレンジ系）
COLOR_SINGLE = "#3498DB"  # 単色グラフ用（青系）


class VisualizationGenerator:
    """RAG評価結果の可視化を行うクラス"""

    def __init__(self):
        """初期化"""
        self.summary_df = None
        self.results_dfs = {}

    def _load_summary_data(self) -> pd.DataFrame:
        """summary.xlsxを読み込む"""
        if not SUMMARY_FILE.exists():
            raise FileNotFoundError(f"summary.xlsxが見つかりません: {SUMMARY_FILE}")
        return pd.read_excel(SUMMARY_FILE)

    def _load_results_data(self) -> dict[str, pd.DataFrame]:
        """全てのresults_{model_name}.xlsxを読み込む（Gemini + Ollama）"""
        results = {}

        # Geminiの結果を読み込む
        if GEMINI_RESULTS_DIR.exists():
            for file_path in GEMINI_RESULTS_DIR.glob("results_*.xlsx"):
                model_name = file_path.stem.replace("results_", "")
                results[model_name] = pd.read_excel(file_path)

        # Ollamaの結果を読み込む
        if OLLAMA_RESULTS_DIR.exists():
            for file_path in OLLAMA_RESULTS_DIR.glob("results_*.xlsx"):
                model_name = file_path.stem.replace("results_", "")
                results[model_name] = pd.read_excel(file_path)

        return results

    def _parse_time_hhmm(self, time_str: str) -> float:
        """hh:mm形式の時間文字列を秒に変換する"""
        if pd.isna(time_str):
            return 0.0
        parts = str(time_str).split(":")
        if len(parts) == 2:
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours * 3600 + minutes * 60
        return 0.0

    def _prepare_summary_for_plotting(self) -> pd.DataFrame:
        """summary.xlsxのデータをグラフ用に整形する"""
        df = self.summary_df.copy()

        # hh:mm形式の実行時間を秒に変換し、eval_nで割って平均を計算
        df["llm.avg_execution_time"] = df.apply(
            lambda row: self._parse_time_hhmm(row["llm.total_execution_time"])
            / row["eval_n"]
            if row["eval_n"] > 0
            else 0,
            axis=1,
        )
        df["rag.avg_execution_time"] = df.apply(
            lambda row: self._parse_time_hhmm(row["rag.total_execution_time"])
            / row["eval_n"]
            if row["eval_n"] > 0
            else 0,
            axis=1,
        )

        return df

    def _setup_plot_style(self):
        """グラフのスタイル設定"""
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10

    def _add_data_labels(self, ax, bars, fmt=".2f"):
        """棒グラフにデータラベルを追加"""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:{fmt}}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    def _create_grouped_bar_chart(
        self,
        df: pd.DataFrame,
        col_no_rag: str,
        col_rag: str,
        title: str,
        ylabel: str,
        filename: str,
    ):
        """グルーピング棒グラフを作成（RAGの有無で比較）"""
        fig, ax = plt.subplots(figsize=(10, 6))

        models = df["llm"].tolist()
        x = range(len(models))
        width = 0.35

        bars1 = ax.bar(
            [i - width / 2 for i in x],
            df[col_no_rag],
            width,
            label="Without RAG",
            color=COLOR_NO_RAG,
        )
        bars2 = ax.bar(
            [i + width / 2 for i in x],
            df[col_rag],
            width,
            label="With RAG",
            color=COLOR_RAG,
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()

        self._add_data_labels(ax, bars1)
        self._add_data_labels(ax, bars2)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close()

    def _create_single_bar_chart(
        self,
        df: pd.DataFrame,
        col: str,
        title: str,
        ylabel: str,
        filename: str,
        ascending: bool = True,
    ):
        """単一棒グラフを作成"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # ソート
        sorted_df = df.sort_values(by=col, ascending=ascending)
        models = sorted_df["llm"].tolist()
        values = sorted_df[col].tolist()

        bars = ax.bar(models, values, color=COLOR_SINGLE)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(models, rotation=45, ha="right")

        self._add_data_labels(ax, bars)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close()

    def _create_grouped_boxplot(
        self,
        col_no_rag: str,
        col_rag: str,
        title: str,
        ylabel: str,
        filename: str,
    ):
        """グルーピング箱ひげ図を作成（RAGの有無で比較）"""
        fig, ax = plt.subplots(figsize=(12, 6))

        models = list(self.results_dfs.keys())
        positions = []
        data_no_rag = []
        data_rag = []
        labels = []

        for i, model in enumerate(models):
            df = self.results_dfs[model]
            pos_base = i * 3
            positions.append(pos_base)
            positions.append(pos_base + 1)
            data_no_rag.append(df[col_no_rag].dropna().tolist())
            data_rag.append(df[col_rag].dropna().tolist())
            labels.append(model)

        # 箱ひげ図の作成
        bp1 = ax.boxplot(
            data_no_rag,
            positions=[i * 3 for i in range(len(models))],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=COLOR_NO_RAG),
            medianprops=dict(color="white"),
            showmeans=True,
            meanprops=dict(marker="x", markerfacecolor="white", markeredgecolor="white"),
        )
        bp2 = ax.boxplot(
            data_rag,
            positions=[i * 3 + 1 for i in range(len(models))],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=COLOR_RAG),
            medianprops=dict(color="white"),
            showmeans=True,
            meanprops=dict(marker="x", markerfacecolor="white", markeredgecolor="white"),
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(models))])
        ax.set_xticklabels(labels, rotation=45, ha="right")

        # 凡例
        ax.legend(
            [bp1["boxes"][0], bp2["boxes"][0]],
            ["Without RAG", "With RAG"],
            loc="upper right",
        )

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close()

    def _create_single_boxplot(
        self,
        col: str,
        title: str,
        ylabel: str,
        filename: str,
        ascending: bool = True,
    ):
        """単一箱ひげ図を作成"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 中央値でソート
        medians = {}
        for model, df in self.results_dfs.items():
            medians[model] = df[col].dropna().median()

        sorted_models = sorted(medians.keys(), key=lambda x: medians[x], reverse=not ascending)

        data = []
        labels = []
        for model in sorted_models:
            df = self.results_dfs[model]
            data.append(df[col].dropna().tolist())
            labels.append(model)

        bp = ax.boxplot(
            data,
            patch_artist=True,
            boxprops=dict(facecolor=COLOR_SINGLE),
            medianprops=dict(color="white"),
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(labels, rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close()

    def generate_execution_time_comparison(self):
        """1. 平均実行時間の比較（棒グラフ）"""
        plot_df = self._prepare_summary_for_plotting()
        self._create_grouped_bar_chart(
            df=plot_df,
            col_no_rag="llm.avg_execution_time",
            col_rag="rag.avg_execution_time",
            title="Execution Time Comparison: Model vs RAG Condition",
            ylabel="Execution Time (seconds)",
            filename="execution_time_comparison.png",
        )

    def generate_noise_sensitivity_comparison(self):
        """2. 平均Noise Sensitivity比較（棒グラフ）"""
        self._create_single_bar_chart(
            df=self.summary_df,
            col="rag.avg_noise_sensitivity",
            title="Noise Sensitivity Comparison by Model",
            ylabel="Noise Sensitivity Score (0.0 - 1.0)",
            filename="noise_sensitivity_model_comparison.png",
            ascending=True,
        )

    def generate_response_relevancy_comparison(self):
        """3. 平均Response Relevancy比較（棒グラフ）"""
        self._create_grouped_bar_chart(
            df=self.summary_df,
            col_no_rag="llm.avg_response_relevancy",
            col_rag="rag.avg_response_relevancy",
            title="Response Relevancy Score: Model vs RAG Condition",
            ylabel="Response Relevancy Score (0.0 - 1.0)",
            filename="response_relevancy_comparison.png",
        )

    def generate_faithfulness_comparison(self):
        """4. 平均Faithfulness比較（棒グラフ）"""
        self._create_single_bar_chart(
            df=self.summary_df,
            col="rag.avg_faithfulness",
            title="Faithfulness Score Comparison by Model",
            ylabel="Faithfulness Score (0.0 - 1.0)",
            filename="faithfulness_model_comparison.png",
            ascending=False,
        )

    def generate_execution_time_boxplot(self):
        """5. 実行時間の分布比較（箱ひげ図）"""
        self._create_grouped_boxplot(
            col_no_rag="llm.execution_time",
            col_rag="rag.execution_time",
            title="Execution Time Distribution: Model vs RAG Condition",
            ylabel="Execution Time (seconds)",
            filename="execution_time_boxplot.png",
        )

    def generate_noise_sensitivity_boxplot(self):
        """6. Noise Sensitivityの分布（箱ひげ図）"""
        self._create_single_boxplot(
            col="rag.noise_sensitivity",
            title="Noise Sensitivity Score Distribution by Model",
            ylabel="Noise Sensitivity Score (0.0 - 1.0)",
            filename="noise_sensitivity_boxplot.png",
            ascending=True,
        )

    def generate_response_relevancy_boxplot(self):
        """7. Response Relevancyの分布比較（箱ひげ図）"""
        self._create_grouped_boxplot(
            col_no_rag="llm.response_relevancy",
            col_rag="rag.response_relevancy",
            title="Response Relevancy Score Distribution: Model vs RAG Condition",
            ylabel="Response Relevancy Score (0.0 - 1.0)",
            filename="response_relevancy_boxplot.png",
        )

    def generate_faithfulness_boxplot(self):
        """8. Faithfulnessの分布（箱ひげ図）"""
        self._create_single_boxplot(
            col="rag.faithfulness",
            title="Faithfulness Score Distribution by Model",
            ylabel="Faithfulness Score (0.0 - 1.0)",
            filename="faithfulness_boxplot.png",
            ascending=False,
        )

    def generate_all_visualizations(self):
        """全てのグラフを生成する"""
        print("=== 可視化スクリプト ===\n")

        # 出力ディレクトリの作成
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # データの読み込み
        print("1. データの読み込み")
        self.summary_df = self._load_summary_data()
        print(f"   - summary.xlsx: {len(self.summary_df)}件のモデル")

        self.results_dfs = self._load_results_data()
        print(f"   - results_*.xlsx: {len(self.results_dfs)}ファイル")
        for model, df in self.results_dfs.items():
            print(f"     - {model}: {len(df)}件")

        # スタイル設定
        self._setup_plot_style()

        # グラフの生成
        print("\n2. グラフの生成")

        print("   - execution_time_comparison.png")
        self.generate_execution_time_comparison()

        print("   - noise_sensitivity_model_comparison.png")
        self.generate_noise_sensitivity_comparison()

        print("   - response_relevancy_comparison.png")
        self.generate_response_relevancy_comparison()

        print("   - faithfulness_model_comparison.png")
        self.generate_faithfulness_comparison()

        print("   - execution_time_boxplot.png")
        self.generate_execution_time_boxplot()

        print("   - noise_sensitivity_boxplot.png")
        self.generate_noise_sensitivity_boxplot()

        print("   - response_relevancy_boxplot.png")
        self.generate_response_relevancy_boxplot()

        print("   - faithfulness_boxplot.png")
        self.generate_faithfulness_boxplot()

        print(f"\n=== 完了: {OUTPUT_DIR} に保存しました ===")


def main():
    """メイン関数"""
    generator = VisualizationGenerator()
    generator.generate_all_visualizations()


if __name__ == "__main__":
    main()
