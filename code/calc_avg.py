import json
from collections import defaultdict
import os

files = [
    "mamba-results/full_results_base.json",
    "mamba-results/full_results_curriculum.json",
    "mamba-results/full_results_random.json",
]

for file in files:
    # Load JSON
    with open(file, "r") as f:
        data = json.load(f)

    # Per-benchmark aggregations
    agg = defaultdict(lambda: {
        "context_length_sum": 0.0,
        "f1_sum": 0.0,
        "precision_sum": 0.0,
        "recall_sum": 0.0,
        "count": 0,
        "total_sum": 0,
    })

    # Global (across all benchmarks) aggregations
    global_stats = {
        "context_length_sum": 0.0,
        "f1_sum": 0.0,
        "precision_sum": 0.0,
        "recall_sum": 0.0,
        "count": 0,
        "total_sum": 0,
    }

    # Aggregate
    for row in data:
        b = row["benchmark"]

        # Per-benchmark
        agg[b]["context_length_sum"] += row["context_length"]
        agg[b]["f1_sum"] += row["f1"]
        agg[b]["precision_sum"] += row["precision"]
        agg[b]["recall_sum"] += row["recall"]
        agg[b]["total_sum"] += row.get("total", 0)
        agg[b]["count"] += 1

        # Global
        global_stats["context_length_sum"] += row["context_length"]
        global_stats["f1_sum"] += row["f1"]
        global_stats["precision_sum"] += row["precision"]
        global_stats["recall_sum"] += row["recall"]
        global_stats["total_sum"] += row.get("total", 0)
        global_stats["count"] += 1

    # Prepare JSONL output file
    out_file = file.replace(".json", ".jsonl").replace("full", "summary")
    out_path = os.path.join("mamba-results", os.path.basename(out_file))

    with open(out_path, "w") as out:
        # Per-benchmark JSONL lines
        for benchmark, vals in agg.items():
            c = vals["count"]
            obj = {
                "benchmark": benchmark,
                "avg_context_length": vals["context_length_sum"] / c,
                "avg_f1": vals["f1_sum"] / c,
                "avg_precision": vals["precision_sum"] / c,
                "avg_recall": vals["recall_sum"] / c,
                "total_examples": vals["total_sum"],
                "num_entries": c,
            }
            out.write(json.dumps(obj) + "\n")

        # Global summary (micro average across all entries)
        gc = global_stats["count"]
        global_obj = {
            "benchmark": "ALL",
            "avg_context_length": global_stats["context_length_sum"] / gc,
            "avg_f1": global_stats["f1_sum"] / gc,
            "avg_precision": global_stats["precision_sum"] / gc,
            "avg_recall": global_stats["recall_sum"] / gc,
            "total_examples": global_stats["total_sum"],
            "num_entries": gc,
        }
        out.write(json.dumps(global_obj) + "\n")

    print(f"Wrote JSONL: {out_path}")
