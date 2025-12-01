import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

def load_json_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

mamba_base = load_json_file('mamba-results/full_results_base.json')
mamba_curr = load_json_file('mamba-results/full_results_curriculum.json')
mamba_rand = load_json_file('mamba-results/full_results_random.json')

mbase_df = pd.DataFrame(mamba_base)
mcurr_df = pd.DataFrame(mamba_curr)
mrand_df = pd.DataFrame(mamba_rand)

mbase_df['model'] = 'Mamba Base'
mcurr_df['model'] = 'Mamba Curr'
mrand_df['model'] = 'Mamba Rand'


mamba_df = pd.concat([mbase_df, mcurr_df, mrand_df], ignore_index=True)


bin_size = 1500
bins = np.arange(0, mamba_df['context_length'].max() + bin_size, bin_size)
mamba_df['context_bin'] = pd.cut(mamba_df['context_length'], bins=bins)

df_grouped = mamba_df.groupby(['context_bin', 'model'])['f1'].mean().reset_index()

df_grouped['bin_midpoint'] = df_grouped['context_bin'].apply(lambda x: x.mid)

df_pivot = df_grouped.pivot(index='bin_midpoint', columns='model', values='f1')

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(df_pivot.index))
width = 0.25

models = ['Mamba Base', 'Mamba Curr', 'Mamba Rand']
colors = ['#66c2a5', '#fc8d62', '#8da0cb']

for i, model in enumerate(models):
    offset = width * (i - 1)
    ax.bar(x + offset, df_pivot[model], width, label=model, 
           color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Context Length', fontsize=12, fontweight='bold')
ax.set_ylabel('Average F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Average F1 Score by Context Length Across Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{int(mid)}' for mid in df_pivot.index], rotation=45, ha='right')
ax.legend(title='Model', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('mamba_f1_vs_context_histogram.png', dpi=300, bbox_inches='tight')
plt.show()




