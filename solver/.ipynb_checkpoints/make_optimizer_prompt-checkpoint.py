import torch

def make_optimizer(cfg, model, center_criterion, stage=None):
    params = []
    
    # Xác định bộ thông số dựa trên stage
    if stage == 1:
        base_lr = cfg.SOLVER.STAGE1.BASE_LR
        weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
        bias_lr_factor = cfg.SOLVER.STAGE1.get('BIAS_LR_FACTOR', 1.0) # Fallback nếu không có
        weight_decay_bias = cfg.SOLVER.STAGE1.WEIGHT_DECAY_BIAS
        opt_name = cfg.SOLVER.STAGE1.OPTIMIZER_NAME
    elif stage == 2:
        base_lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        bias_lr_factor = cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
        weight_decay_bias = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        opt_name = cfg.SOLVER.STAGE2.OPTIMIZER_NAME
    else:
        # Mặc định (Single stage hoặc flow cũ)
        base_lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        bias_lr_factor = cfg.SOLVER.BIAS_LR_FACTOR
        weight_decay_bias = cfg.SOLVER.WEIGHT_DECAY_BIAS
        opt_name = cfg.SOLVER.OPTIMIZER_NAME

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        
        lr = base_lr
        wd = weight_decay
        if "bias" in key:
            lr = base_lr * bias_lr_factor
            wd = weight_decay_bias
            
        if cfg.get('SOLVER.LARGE_FC_LR', False) or (stage == 2 and cfg.SOLVER.STAGE2.get('LARGE_FC_LR', False)):
            if "classifier" in key or "arcface" in key:
                lr = base_lr * 2
                print(f'Using 2x learning rate for classifier: {key}')

        params += [{"params": [value], "lr": lr, "weight_decay": wd}]

    if opt_name == 'SGD':
        optimizer = getattr(torch.optim, opt_name)(params, momentum=cfg.SOLVER.get('MOMENTUM', 0.9))
    elif opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    elif opt_name == 'Adam':
        optimizer = torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
    else:
        optimizer = getattr(torch.optim, opt_name)(params)
        
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.get('CENTER_LR', 0.5))

    return optimizer, optimizer_center