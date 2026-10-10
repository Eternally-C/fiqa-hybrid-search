from pathlib import Path

import matplotlib

# 使用非交互后端，直接保存图片，无需弹出绘图窗口。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["BM25", "Dense", "Weighted Hybrid"]
DISPLAY_LABELS = ["BM25", "Dense", "Weighted\nHybrid"]
METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]
COLORS = ["#7B8794", "#4C78A8", "#F28E2B"]


def main():
    """从已保存的最终 Test 指标生成比较图。"""

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "results" / "test_baselines.csv"
    output_dir = project_root / "results" / "figures"

    results = pd.read_csv(input_path)

    # 检查结果结构，避免遗漏方法或将重复记录绘制为正常结果。
    expected_columns = ["Method", *METRICS]

    if list(results.columns) != expected_columns:
        raise ValueError(
            f"CSV 列名异常：预期 {expected_columns}，"
            f"实际 {list(results.columns)}。"
        )

    if (
        len(results) != len(METHOD_ORDER)
        or set(results["Method"]) != set(METHOD_ORDER)
    ):
        raise ValueError(
            "CSV 必须恰好包含 BM25、Dense、Weighted Hybrid 各一行。"
        )

    results = results.set_index("Method").loc[METHOD_ORDER]
    metric_values = results[METRICS].to_numpy(dtype=float)

    if (
        not np.isfinite(metric_values).all()
        or np.any((metric_values < 0) | (metric_values > 1))
    ):
        raise ValueError("指标必须是 0 到 1 之间的有限数值。")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
        layout="constrained",
    )

    # Query 数量来自已经审查通过的最终 Test 日志。
    fig.suptitle(
        "FiQA-2018: Final Test Retrieval Performance\n"
        "648 queries. Parameters selected on Validation.",
        fontsize=14,
    )

    positions = np.arange(len(METHOD_ORDER))

    for ax, metric in zip(axes.flat, METRICS):
        scores = results[metric].to_numpy(dtype=float)

        bars = ax.bar(
            positions,
            scores,
            color=COLORS,
            width=0.6,
        )

        ax.set_xticks(
            positions,
            labels=DISPLAY_LABELS,
            fontsize=10,
        )
        ax.set_title(metric, fontsize=12)
        ax.set_ylabel("Score")

        # 统一指标范围，保留零起点，避免夸大小幅差异。
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.bar_label(
            bars,
            labels=[f"{score:.4f}" for score in scores],
            padding=3,
            fontsize=10,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # PNG 用于展示；SVG 保留矢量图，便于缩放和导出。
    for extension in ("png", "svg"):
        output_path = output_dir / f"test_retrieval_metrics.{extension}"

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Saved figure to: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()