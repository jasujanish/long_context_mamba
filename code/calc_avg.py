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

    agg = defaultdict(lambda: {
        "context_length_sum": 0,
        "f1_sum": 0,
        "precision_sum": 0,
        "recall_sum": 0,
        "count": 0,
        "total_sum": 0,
    })

    # Aggregate per benchmark
    for row in data:
        b = row["benchmark"]
        agg[b]["context_length_sum"] += row["context_length"]
        agg[b]["f1_sum"] += row["f1"]
        agg[b]["precision_sum"] += row["precision"]
        agg[b]["recall_sum"] += row["recall"]
        agg[b]["total_sum"] += row.get("total", 0)
        agg[b]["count"] += 1

    # Prepare JSONL output file
    out_file = file.replace(".json", ".jsonl").replace("full", "summary")
    out_path = os.path.join("mamba-results", os.path.basename(out_file))

    with open(out_path, "w") as out:
        # Compute averages and write each as JSONL line
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

    print(f"Wrote JSONL: {out_path}")