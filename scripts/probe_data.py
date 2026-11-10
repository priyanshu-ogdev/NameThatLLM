import pandas as pd, numpy as np

df = pd.read_parquet('data/processed/train_folds.parquet')
g = df.groupby('prompt_id')['model'].agg(['count','nunique'])

print('Distribution of prompt sizes (count of passages per prompt_id):')
print(g['count'].value_counts().sort_index().head(30))

print()
print('Prompts with 18 passages:', len(g[g['count'] == 18]))
print('Prompts with count == nunique (no duplicates):', len(g[g['count'] == g['nunique']]))

clean_all = g[(g['count'] == g['nunique'])]
print('Clean prompt sizes distribution:')
print(clean_all['count'].value_counts().sort_index())

print()
# How does the competition data structure work?
# A "prompt_id" in train = one specific question/prompt, possibly answered by multiple or all 18 models
# Let's count per fold
for fold in df['fold'].unique():
    sub = df[df['fold'] == fold]
    g_fold = sub.groupby('prompt_id')['model'].agg(['count','nunique'])
    g18 = g_fold[g_fold['count'] == 18]
    print(f"Fold {fold}: {len(sub)} rows, {sub['prompt_id'].nunique()} prompts, {len(g18)} prompts with 18 passages")

