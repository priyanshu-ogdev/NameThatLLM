import json

for path in ["notebooks/kaggle_dry_run.ipynb", "notebooks/kaggle_tri_ensemble.ipynb"]:
    nb = json.load(open(path, "r", encoding="utf-8"))
    print(path, "total cells:", len(nb["cells"]))
    for i, c in enumerate(nb["cells"]):
        src_preview = "".join(c["source"])[:60].replace("\n", " ")
        print(f"  cell {i} ({c['cell_type']}): {src_preview}")

