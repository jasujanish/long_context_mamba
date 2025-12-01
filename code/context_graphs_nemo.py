import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

def load_json_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

models = ['base', 'random', 'curriculum']
benchmarks = ['2wikimqa', 'qasper', 'hotpotqa', 'multifieldqa_en']

all_dfs = []

for model in models:
    for benchmark in benchmarks:
        filepath = f'nemotron-results/full_score/nemotron-{model}-{benchmark}.json'
        data = load_json_file(filepath)
        df = pd.DataFrame(data)
        df['model'] = f'Nemotron {model.capitalize()}'
        df['benchmark'] = benchmark
        all_dfs.append(df)
        print(f"Loaded: {filepath} - {len(df)} records")

nemotron_df = pd.concat(all_dfs, ignore_index=True)

bin_size = 5000
bins = np.arange(0, nemotron_df['context_length'].max() + bin_size, bin_size)
nemotron_df['context_bin'] = pd.cut(nemotron_df['context_length'], bins=bins)

df_grouped = nemotron_df.groupby(['context_bin', 'model'])['f1'].mean().reset_index()

df_grouped['bin_midpoint'] = df_grouped['context_bin'].apply(lambda x: x.mid)

df_pivot = df_grouped.pivot(index='bin_midpoint', columns='model', values='f1')

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(df_pivot.index))
width = 0.25

models_display = ['Nemotron Base', 'Nemotron Curriculum', 'Nemotron Random']
colors = ['#66c2a5', '#fc8d62', '#8da0cb']

for i, model in enumerate(models_display):
    offset = width * (i - 1)
    if model in df_pivot.columns:
        ax.bar(x + offset, df_pivot[model], width, label=model, 
               color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Context Length', fontsize=12, fontweight='bold')
ax.set_ylabel('Average F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Average F1 Score by Context Length', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{int(mid)}' for mid in df_pivot.index], rotation=45, ha='right')
ax.legend(title='Model', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('nemotron_f1_vs_context_histogram.png', dpi=300, bbox_inches='tight')
plt.show()
