with open("notebooks/kaggle_tri_ensemble.py", "r", encoding="utf-8") as f:
    text = f.read()

target = """    ws, wd, wq, wb = best_w
    print(f"\\n  Blend weights: CPU={ws} DeBERTa={wd} Qwen={wq} BGE={wb}", flush=True)
    print(f"  OOF QUAD BLEND: {blend_score*100:.2f}%\\n", flush=True)"""

replacement = """    ws, wd, wq, wb = best_w
    print(f"\\n  Blend weights: CPU={ws} DeBERTa={wd} Qwen={wq} BGE={wb}", flush=True)
    print(f"  OOF QUAD BLEND: {blend_score*100:.2f}%\\n", flush=True)

    # Diagnostic: Ridge Stacker Comparison
    try:
        from sklearn.linear_model import RidgeClassifier
        avail_streams = [margins_to_probs(oof_svm, temp=0.20)]
        if oof_deb is not None: avail_streams.append(margins_to_probs(oof_deb, temp=1.2))
        if oof_qwen is not None: avail_streams.append(margins_to_probs(oof_qwen, temp=1.5))
        if oof_bge is not None: avail_streams.append(margins_to_probs(oof_bge, temp=1.2))
        if len(avail_streams) > 1:
            stack_mask = ~np.isnan(avail_streams[0][:, 0])
            for st in avail_streams[1:]:
                stack_mask = stack_mask & ~np.isnan(st[:, 0])
            if np.any(stack_mask):
                lbl2id = {l: i for i, l in enumerate(LABELS)}
                y_stack = np.array([lbl2id[y] for y in y_true[stack_mask]])
                X_oof_stack = np.hstack([st[stack_mask] for st in avail_streams])
                ridge_stacker = RidgeClassifier(alpha=1.0)
                ridge_stacker.fit(X_oof_stack, y_stack)
                ridge_preds = ridge_stacker.predict(X_oof_stack)
                ridge_score = balanced_accuracy_score(y_stack, ridge_preds)
                print(f"  OOF RIDGE STACKER: {ridge_score*100:.2f}% (diagnostic comparison)\\n", flush=True)
    except Exception as e:
        print(f"  [Diagnostic Stacker] Note: {e}", flush=True)"""

text_lf = text.replace("\r\n", "\n")
target_lf = target.replace("\r\n", "\n")
replacement_lf = replacement.replace("\r\n", "\n")

if target_lf in text_lf:
    text_lf = text_lf.replace(target_lf, replacement_lf, 1)
    with open("notebooks/kaggle_tri_ensemble.py", "w", encoding="utf-8") as f:
        f.write(text_lf)
    print("Successfully inserted Ridge Stacker diagnostic block.")
else:
    print("Target not found in text_lf!")

