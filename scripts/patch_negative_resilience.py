import re
from pathlib import Path

def apply_defensive_patches(path_str):
    p = Path(path_str)
    content = p.read_text(encoding="utf-8")

    # 1. Binary classification resilience in DualLinearClassifier
    old_df = """    def decision_function(self, X):
        d_svc = self.svc_.decision_function(X)
        d_ridge = self.ridge_.decision_function(X)
        m = self.svc_weight * d_svc + (1.0 - self.svc_weight) * d_ridge
        if self.canonical_labels is not None:
            return align_margins(m, self._raw_classes, self.canonical_labels)
        return m"""

    new_df = """    def decision_function(self, X):
        d_svc = self.svc_.decision_function(X)
        d_ridge = self.ridge_.decision_function(X)
        if d_svc.ndim == 1:
            d_svc = np.column_stack([-d_svc, d_svc])
        if d_ridge.ndim == 1:
            d_ridge = np.column_stack([-d_ridge, d_ridge])
        m = self.svc_weight * d_svc + (1.0 - self.svc_weight) * d_ridge
        if self.canonical_labels is not None:
            return align_margins(m, self._raw_classes, self.canonical_labels)
        return m"""
    content = content.replace(old_df, new_df)

    # 2. Missing prompt_id resilience in make_prompt_groups
    old_mpg = """def make_prompt_groups(df):
    \"\"\"DSU to ensure identical texts across different prompts never cross folds.\"\"\"
    dsu = DSU(df["prompt_id"].unique())
    seen = {}
    for pid, text in zip(df["prompt_id"], df["text"]):
        h = hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()
        if h in seen:
            dsu.union(pid, seen[h])
        else:
            seen[h] = pid
    return np.array([dsu.find(p) for p in df["prompt_id"]])"""

    new_mpg = """def make_prompt_groups(df):
    \"\"\"DSU to ensure identical texts across different prompts never cross folds.\"\"\"
    prompt_col = df["prompt_id"] if "prompt_id" in df.columns else pd.Series(np.arange(len(df)), index=df.index)
    dsu = DSU(prompt_col.unique())
    seen = {}
    for pid, text in zip(prompt_col, df["text"]):
        h = hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()
        if h in seen:
            dsu.union(pid, seen[h])
        else:
            seen[h] = pid
    return np.array([dsu.find(p) for p in prompt_col])"""
    content = content.replace(old_mpg, new_mpg)

    p.write_text(content, encoding="utf-8")
    print(f"Patched {path_str} successfully.")

apply_defensive_patches("notebooks/kaggle_tri_ensemble.py")
apply_defensive_patches("notebooks/kaggle_dry_run.py")

