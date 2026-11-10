import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/train_folds.parquet')
g = df.groupby('prompt_id')['model'].agg(['count','nunique'])

# Largest clean prompts
clean = g[(g['count'] == g['nunique'])].sort_values('count', ascending=False).head(3)
for pid, row in clean.iterrows():
    sub = df[df['prompt_id']==pid]
    c = row['count']
    print(f"pid {pid}: {c} models: {sorted(sub['model'].unique())}")
print()

# Family breakdown in the 18 competition models
models = sorted(df['model'].unique())
families = {}
for m in models:
    if 'claude' in m:
        families.setdefault('claude', []).append(m)
    elif 'gemini' in m:
        families.setdefault('gemini', []).append(m)
    elif 'gpt' in m:
        families.setdefault('gpt', []).append(m)
    elif 'glm' in m:
        families.setdefault('glm', []).append(m)
    elif 'qwen' in m:
        families.setdefault('qwen', []).append(m)
    elif 'kimi' in m:
        families.setdefault('kimi', []).append(m)

print('Family sizes:')
for fam, ms in sorted(families.items()):
    print(f'  {fam} ({len(ms)}): {ms}')
print()
total = sum(len(v) for v in families.values())
print(f'Total models: {total}')
singletons = [f for f,ms in families.items() if len(ms)==1]
print(f'Singleton families (easiest): {singletons}')
multi = [(f, len(ms)) for f,ms in families.items() if len(ms)>1]
print(f'Multi-model families (hardest): {multi}')

# Data coverage per model
print('\nTraining data per model:')
print(df['model'].value_counts().sort_values(ascending=False).to_string())

