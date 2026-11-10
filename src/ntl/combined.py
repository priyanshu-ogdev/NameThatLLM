"""Fixed CPU/Qwen rank blend: no fitted ensemble weights or calibration."""
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from ntl.generative import FOLD_SHA256, check_frame, discover_data, discover_model, extract_frozen, load_frozen, make_head, official_folds, sha256_file, write_json, write_submission
from ntl.models import advanced_features


def build_cpu():
    features = advanced_features('multiview_svm')
    features.set_params(transformer_weights={'style': 0.2})
    return Pipeline([('features', features), ('classifier', LinearSVC(
        C=1.5, class_weight='balanced', dual=True, max_iter=15000, random_state=42))])


def rank_blend(cpu_scores, qwen_scores):
    """Use the exact label-free equal-rank rule evaluated on retained OOF scores."""
    cpu_scores, qwen_scores = np.asarray(cpu_scores), np.asarray(qwen_scores)
    if (cpu_scores.ndim != 2 or cpu_scores.shape != qwen_scores.shape
            or cpu_scores.shape[1] < 2 or not np.isfinite(cpu_scores).all()
            or not np.isfinite(qwen_scores).all()):
        raise ValueError('Invalid or misaligned decision score matrices')
    ranks = rankdata(cpu_scores, axis=1) + rankdata(qwen_scores, axis=1)
    tie_break = (cpu_scores - cpu_scores.min(axis=1, keepdims=True)) / np.maximum(
        np.ptp(cpu_scores, axis=1, keepdims=True), 1e-12)
    return (ranks + 1e-6 * tie_break).argmax(axis=1)


def ordered_scores(model, values, labels):
    if set(model.classes_) != set(labels):
        raise ValueError('Classifier classes differ from competition labels')
    order = [list(model.classes_).index(label) for label in labels]
    return model.decision_function(values)[:, order]


def fit_cpu(texts, labels):
    model = build_cpu()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        with threadpool_limits(limits=2):
            model.fit(texts, labels)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError('CPU SVM did not converge; share this message before submitting.')
    return model


def predict_combined(cpu, qwen, frame, features, labels, batch_size=256):
    """Bound text/vectorizer memory; keep scores in one explicit class order."""
    parts = []
    with threadpool_limits(limits=2):
        for start in range(0, len(frame), batch_size):
            end = start + batch_size
            left = ordered_scores(cpu, frame.text.iloc[start:end], labels)
            right = ordered_scores(qwen, features[start:end, 0], labels)
            parts.append(labels[rank_blend(left, right)])
    return np.concatenate(parts)


def evaluate_combined(train, features, folds, labels, out):
    y = train.model.to_numpy()
    cpu_predictions = np.empty(len(train), dtype=object)
    qwen_predictions = np.empty(len(train), dtype=object)
    combined_predictions = np.empty(len(train), dtype=object)
    cpu_margins = np.zeros((len(train), len(labels)), dtype=np.float64)
    qwen_margins = np.zeros_like(cpu_margins)
    fold_results = []
    for fold in sorted(set(folds)):
        tr, va = folds != fold, folds == fold
        if (set(train.prompt_id[tr]) & set(train.prompt_id[va])
                or set(y[tr]) != set(labels) or set(y[va]) != set(labels)):
            raise ValueError('Invalid grouped validation folds')
        cpu = fit_cpu(train.text[tr], y[tr])
        head = make_head('linear')
        with threadpool_limits(limits=2):
            head.fit(features[tr, 0], y[tr])
            cpu_margins[va] = ordered_scores(cpu, train.text[va], labels)
            qwen_margins[va] = ordered_scores(head, features[va, 0], labels)
            cpu_predictions[va] = cpu.predict(train.text[va])
            qwen_predictions[va] = head.predict(features[va, 0])
        combined_predictions[va] = labels[rank_blend(cpu_margins[va], qwen_margins[va])]
        row = {'fold': int(fold), **{name: float(balanced_accuracy_score(y[va], pred[va]))
               for name, pred in [('cpu', cpu_predictions), ('qwen', qwen_predictions), ('combined', combined_predictions)]}}
        fold_results.append(row)
        print('Validation fold:', row, flush=True)
    metrics = {name: float(balanced_accuracy_score(y, pred)) for name, pred in
               [('cpu', cpu_predictions), ('qwen', qwen_predictions), ('combined', combined_predictions)]}
    metrics['qwen_correct_on_cpu_errors'] = int(sum((qwen_predictions == y) & (cpu_predictions != y)))
    metrics['folds'] = fold_results
    metrics['note'] = 'Fixed blend on original reused prompt folds; selected base configurations; not an independent holdout or leaderboard score.'
    pd.DataFrame({'id': train.id, 'fold': folds, 'truth': y, 'cpu': cpu_predictions,
                  'qwen': qwen_predictions, 'combined': combined_predictions}).to_csv(out / 'oof.csv', index=False)
    np.savez_compressed(out / 'oof_scores.npz', cpu=cpu_margins, qwen=qwen_margins, labels=labels)
    pd.DataFrame({'model': labels, 'recall': recall_score(y, combined_predictions, labels=labels, average=None)}).to_csv(out / 'per_class.csv', index=False)
    pd.DataFrame(confusion_matrix(y, combined_predictions, labels=labels), index=labels, columns=labels).to_csv(out / 'confusion.csv')
    write_json(out / 'validation.json', metrics)
    return metrics


def run_combined(input_root='/kaggle/input', working='/kaggle/working', allow_cpu=False, run_cv=True,
                 model_loader=None, model_discovery=None, feature_extractor=None,
                 experiment_name='combined-v1', cache_name='qwen-frozen-v1/cache'):
    started = time.perf_counter()
    out = Path(working) / experiment_name
    out.mkdir(parents=True, exist_ok=True)
    data = discover_data(input_root)
    train = pd.read_csv(data / 'train.csv', dtype=str, keep_default_na=False)
    test = pd.read_csv(data / 'test.csv', dtype=str, keep_default_na=False)
    check_frame(train, ['id', 'prompt_id', 'text', 'model'])
    check_frame(test, ['id', 'text'])
    folds = official_folds(train, data / 'train.csv')
    labels = np.array(sorted(train.model.unique()))
    if len(labels) != 18:
        raise ValueError('Expected 18 labels')
    print('Data and original folds verified:', data, flush=True)
    model_loader = model_loader or load_frozen
    model_discovery = model_discovery or discover_model
    feature_extractor = feature_extractor or extract_frozen
    model, tokenizer, provenance = model_loader(model_discovery(input_root), allow_cpu=allow_cpu)
    # Same cache location and extraction settings as the frozen experiment.
    cache = Path(working) / cache_name
    features, train_info = feature_extractor(train, model, tokenizer, provenance, cache)
    validation = evaluate_combined(train, features, folds, labels, out) if run_cv else None
    cpu = fit_cpu(train.text, train.model)
    head = make_head('linear')
    with threadpool_limits(limits=2):
        head.fit(features[:, 0], train.model)
    representative = np.arange(4000) % len(train)
    benchmark_started = time.perf_counter()
    predict_combined(cpu, head, train.iloc[representative].reset_index(drop=True), features[representative], labels)
    benchmark_seconds = time.perf_counter() - benchmark_started
    print(f'Combined classifier 4000-row benchmark: {benchmark_seconds:.1f} seconds; excludes transformer extraction, not an accuracy test.', flush=True)
    joblib.dump(cpu, out / 'cpu.joblib')
    joblib.dump(head, out / 'qwen_head.joblib')
    test_features, test_info = feature_extractor(test, model, tokenizer, provenance, cache)
    predictions = predict_combined(cpu, head, test, test_features, labels)
    target = write_submission(test, predictions, labels, working)
    report = {'validation': validation, 'provenance': provenance, 'labels': labels.tolist(),
              'training_sha256': sha256_file(data / 'train.csv'), 'fold_sha256': FOLD_SHA256,
              'test_sha256': sha256_file(data / 'test.csv'), 'test_rows': len(test),
              'train_extraction': train_info, 'test_extraction': test_info,
              'historical_qwen_correct_on_cpu_errors': 417,
              'historical_fixed_blend_oof': 0.4093263733532189,
              'combined_classifier_4000_seconds': benchmark_seconds,
              'rule': 'Equal ranks, CPU decision score breaks ties; Qwen middle_mean_linear; no ensemble fitting.',
              'total_seconds': time.perf_counter() - started}
    write_json(out / 'summary.json', report)
    print('RESULT TO SHARE:', {'validation': validation, 'test_rows': len(test), 'elapsed_minutes': report['total_seconds'] / 60}, flush=True)
    print(f'SUCCESS: {target} with {len(test)} rows. Combined model run complete.', flush=True)
    return report
