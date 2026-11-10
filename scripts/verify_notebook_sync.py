import json

nb_dry = json.load(open("notebooks/kaggle_dry_run.ipynb", "r", encoding="utf-8"))
src_dry = "".join("".join(c["source"]) for c in nb_dry["cells"] if c["cell_type"] == "code")
print("kaggle_dry_run.ipynb:")
print("  Has Concurrent Check:", "[Concurrent Check]" in src_dry)
print("  Has Check 4/7:", "[Check 4/7]" in src_dry)
print("  Has Check 7/7:", "[Check 7/7]" in src_dry)
print("  Code length:", len(src_dry))

nb_tri = json.load(open("notebooks/kaggle_tri_ensemble.ipynb", "r", encoding="utf-8"))
src_tri = "".join("".join(c["source"]) for c in nb_tri["cells"] if c["cell_type"] == "code")
print("\nkaggle_tri_ensemble.ipynb:")
print("  Has ThreadPoolExecutor:", "ThreadPoolExecutor(max_workers=2)" in src_tri)
print("  Has Sinkhorn-Knopp:", "sinkhorn_knopp_balance" in src_tri)
print("  Code length:", len(src_tri))
