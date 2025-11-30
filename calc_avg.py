import json
from collections import defaultdict

files = ["full_results_curriculum.json", "full_results_random.json"]
for file in files:
    # Setup
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

    # Aggregate
    for row in data:
        b = row["benchmark"]
        agg[b]["context_length_sum"] += row["context_length"]
        agg[b]["f1_sum"] += row["f1"]
        agg[b]["precision_sum"] += row["precision"]
        agg[b]["recall_sum"] += row["recall"]
        agg[b]["total_sum"] += row.get("total", 0)
        agg[b]["count"] += 1

    # Compute averages
    results = {}
    for benchmark, vals in agg.items():
        c = vals["count"]
        results[benchmark] = {
            "avg_context_length": vals["context_length_sum"] / c,
            "avg_f1": vals["f1_sum"] / c,
            "avg_precision": vals["precision_sum"] / c,
            "avg_recall": vals["recall_sum"] / c,
            "total_examples": vals["total_sum"],
            "num_entries": c,
        }

    # Pretty print
    print(f'Results for {file}')
    print(json.dumps(results, indent=4))
    print('-'*50)