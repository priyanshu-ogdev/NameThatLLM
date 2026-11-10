import sys
import torch
import torch.nn as nn

# Let's inspect HuggingFace transformers DebertaV2 architecture if installed, or build mock
try:
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification, BertConfig, BertForSequenceClassification
    deb_cfg = DebertaV2Config(num_labels=18, hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=256)
    deb_model = DebertaV2ForSequenceClassification(deb_cfg)
    
    # Run get_optimizer_grouped_parameters from kaggle_tri_ensemble
    sys.path.insert(0, ".")
    from notebooks.kaggle_tri_ensemble import get_optimizer_grouped_parameters
    
    opt_params = get_optimizer_grouped_parameters(deb_model, lr_head=1e-4, lr_backbone=2e-5)
    
    all_named_params = dict(deb_model.named_parameters())
    grouped_params_set = set()
    for g in opt_params:
        for p in g["params"]:
            grouped_params_set.add(id(p))
            
    missing_params = [n for n, p in all_named_params.items() if id(p) not in grouped_params_set]
    print("DeBERTa missing params count:", len(missing_params))
    for m in missing_params:
        print("  Missing param:", m)
        
    bge_cfg = BertConfig(num_labels=18, hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=256)
    bge_model = BertForSequenceClassification(bge_cfg)
    opt_params_bge = get_optimizer_grouped_parameters(bge_model, lr_head=1e-4, lr_backbone=2e-5)
    grouped_bge_set = set()
    for g in opt_params_bge:
        for p in g["params"]:
            grouped_bge_set.add(id(p))
    missing_bge = [n for n, p in bge_model.named_parameters() if id(p) not in grouped_bge_set]
    print("BGE missing params count:", len(missing_bge))
    for m in missing_bge:
        print("  Missing BGE param:", m)
        
except Exception as e:
    print("Exception during test:", e)

