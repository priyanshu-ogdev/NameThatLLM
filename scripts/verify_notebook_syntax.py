import ast
import sys

files = ['notebooks/kaggle_tri_ensemble.py', 'notebooks/kaggle_dry_run.py']
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            source = fp.read()
        ast.parse(source, filename=f)
        print(f"[OK] {f} parsed cleanly as valid Python AST.")
    except Exception as e:
        print(f"[ERROR] {f}: {e}")
        sys.exit(1)

print("All files parsed successfully.")

