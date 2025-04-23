import torch

def load_pretrained(cfg, model, logger=None, phase="train", strict=True):
        
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
    
    if logger is not None:
        logger.info(f"Loading pretrain model from {ckpt_path}")
    
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=strict)
    return model


def load_pretrained_vae(cfg, model, logger=None):
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE,
                            map_location="cpu")['state_dict']
    if logger is not None:
        logger.info(f"Loading pretrain vae from {cfg.TRAIN.PRETRAINED_VAE}")
        
    # Extract encoder/decoder
    from collections import OrderedDict
    for vae_type in ['vae_body', 'vae_hand', 'vae_transform']:
        vae_dict = OrderedDict()
        for k, v in state_dict.items():
            if vae_type in k:
                name = k.replace(vae_type + ".", "")
                vae_dict[name] = v
        if getattr(model, vae_type, None) is not None:
            getattr(model, vae_type).load_state_dict(vae_dict, strict=True)
            # model.vae.load_state_dict(vae_dict, strict=True)
    
    return model
