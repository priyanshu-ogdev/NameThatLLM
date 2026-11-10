"""Real-weight CPU integration check; not a new competition evaluation."""
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from ntl.combined import fit_cpu, make_head, predict_combined, rank_blend
from ntl.generative import extract_frozen, load_frozen, write_json, write_submission


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    train = pd.read_csv(root / 'data/official/train.csv', dtype=str, keep_default_na=False)
    labels = np.array(sorted(train.model.unique()))
    qwen_oof = pd.read_csv(root / 'runs/qwen-kaggle-v1-received-20261023/oof.csv', dtype={'id': str})
    qwen_scores = np.load(root / 'runs/qwen-kaggle-v1-received-20261023/decision_scores.npy', allow_pickle=False)
    with np.load(root / 'runs/linear-midstyle-v1/multi_c150_s020/oof_scores.npz', allow_pickle=False) as saved:
        order = [saved['labels'].tolist().index(label) for label in labels]
        cpu_scores = saved['scores'][:, order]
    cpu_oof = pd.read_csv(root / 'runs/linear-midstyle-v1/multi_c150_s020/oof.csv', dtype={'id': str})
    assert cpu_oof.id.tolist() == qwen_oof.id.tolist() == train.id.tolist()
    blend = labels[rank_blend(cpu_scores, qwen_scores)]
    score = balanced_accuracy_score(train.model, blend)
    assert abs(score - 0.4093263733532189) < 1e-12
    unique_help = int(sum((qwen_oof.prediction == qwen_oof.truth) & (cpu_oof.prediction != qwen_oof.truth)))
    assert unique_help == 417
    # Full-data CPU fit; actual pretrained transformer on one passage per class.
    cpu = fit_cpu(train.text, train.model)
    small = train.groupby('model', sort=True).head(1).reset_index(drop=True)
    model, tokenizer, provenance = load_frozen('/workspace/models/Qwen3-1.7B-Base-kaggle-v1', allow_cpu=True)
    with tempfile.TemporaryDirectory() as directory:
        features, info = extract_frozen(small, model, tokenizer, provenance, directory)
        head = make_head('linear')
        with threadpool_limits(limits=2):
            head.fit(features[:, 0], small.model)
        repeated = np.arange(4000) % len(small)
        frame = small.iloc[repeated].reset_index(drop=True).copy()
        frame['id'] = [f'{i:08d}' for i in range(4000)]
        before = time.perf_counter()
        predictions = predict_combined(cpu, head, frame, features[repeated], labels)
        seconds = time.perf_counter() - before
        path = write_submission(frame, predictions, labels, directory)
        saved = pd.read_parquet(path)
        assert len(saved) == 4000 and saved.id.tolist() == frame.id.tolist()
        assert saved.model.isin(labels).all()
    write_json(root / 'reports/combined-v1/local_validation.json', {
        'historical_oof_score_reproduced_by_production_rule': float(score),
        'historical_qwen_correct_on_cpu_errors_verified': unique_help,
        'real_frozen_transformer_rows': len(small), 'cpu_fit_rows': len(train),
        'submission_roundtrip_rows': 4000, 'combined_prediction_seconds': seconds,
        'total_seconds': time.perf_counter() - started,
        'provenance': provenance,
        'limitations': 'CPU functional check. Qwen head fits 18 real passages, not all 2699; 4000 rows repeat their cached representations. Timing excludes extraction and understates full-head cost. No new competition score, full GPU run or hidden-test timing is claimed.'})
    print('PASS: exact historical score, 417 complementary answers, real-weight extraction, full CPU fit and 4000-row Parquet round trip.')


if __name__ == '__main__':
    main()
