with open('notebooks/kaggle_tri_ensemble.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_triple_double = False
in_triple_single = False

for i, line in enumerate(lines[:1100], 1):
    # count triple quotes
    c_td = line.count('"""')
    c_ts = line.count("'''")
    if c_td % 2 != 0:
        in_triple_double = not in_triple_double
        print(f"Line {i} toggled triple_double -> {in_triple_double}: {line.strip()[:60]}")
    if c_ts % 2 != 0:
        in_triple_single = not in_triple_single
        print(f"Line {i} toggled triple_single -> {in_triple_single}: {line.strip()[:60]}")

