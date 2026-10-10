from pathlib import Path

import matplotlib

# 使用非交互后端，直接保存图片，不打开图形窗口。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_FILES = {
    "validation_baselines.csv": [
        "BM25",
        "Dense",
        "Weighted Hybrid",
    ],
    "reranker_validation.csv": [
        "Weighted Hybrid + Cross-Encoder",
    ],
    "dense_reranker_validation.csv": [
        "Dense + Cross-Encoder",
    ],
    "passage_reranker_validation.csv": [
        "Weighted Hybrid + Passage Cross-Encoder",
    ],
}

METHOD_ORDER = [
    "BM25",
    "Dense",
    "Weighted Hybrid",
    "Weighted Hybrid + Cross-Encoder",
    "Dense + Cross-Encoder",
    "Weighted Hybrid + Passage Cross-Encoder",
]

METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]

COLORS = [
    "#7B8794",
    "#4C78A8",
    "#F28E2B",
    "#9B8BC1",
    "#9B8BC1",
    "#9B8BC1",
]


def main():
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    output_dir = results_dir / "figures"

    expected_columns = ["Method", *METRICS]
    frames = []

    # 只读取已有实验结果，不加载模型、数据集或任何 qrels。
    for filename, expected_methods in RESULT_FILES.items():
        input_path = results_dir / filename
        frame = pd.read_csv(input_path)

        if list(frame.columns) != expected_columns:
            raise ValueError(f"{filename}: CSV columns do not match.")

        if (
            len(frame) != len(expected_methods)
            or set(frame["Method"]) != set(expected_methods)
        ):
            raise ValueError(f"{filename}: method names or row count do not match.")

        frames.append(frame)

    # 使用固定展示顺序，不重新排序或选择实验配置。
    results = pd.concat(frames, ignore_index=True)
    results = results.set_index("Method").loc[METHOD_ORDER]

    metric_values = results[METRICS].to_numpy(dtype=float)

    if (
        not np.isfinite(metric_values).all()
        or np.any((metric_values < 0) | (metric_values > 1))
    ):
        raise ValueError("Metric values must be finite numbers between 0 and 1.")

    scores = results["nDCG@10"].to_numpy(dtype=float)
    positions = np.arange(len(METHOD_ORDER))

    fig, ax = plt.subplots(
        figsize=(12, 6),
        layout="constrained",
    )

    bars = ax.barh(
        positions,
        scores,
        color=COLORS,
        height=0.6,
    )

    ax.set_yticks(positions, labels=METHOD_ORDER, fontsize=10)
    ax.invert_yaxis()

    ax.set_title(
        "FiQA-2018: Validation Retrieval and Reranking\n"
        "500 queries. Model selection metric: nDCG@10.",
        fontsize=14,
        pad=14,
    )

    ax.set_xlabel("nDCG@10", fontsize=11)

    # 与 Test 图保持相同的 0–1 指标范围，避免放大微小差异。
    ax.set_xlim(0, 1)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.bar_label(
        bars,
        labels=[f"{score:.4f}" for score in scores],
        padding=4,
        fontsize=10,
    )

    # PNG 用于直接展示，SVG 用于清晰缩放；不修改原始 CSV。
    output_dir.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "svg"):
        output_path = output_dir / f"validation_ndcg_comparison.{extension}"
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Saved: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()