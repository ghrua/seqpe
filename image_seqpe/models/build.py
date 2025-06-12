# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin import SwinTransformer
from .resnet import ResNet50
from .vit_timm import (
    vit_tiny_patch16_224,
    vit_small_patch16_224,
    vit_small_patch16_384,
    vit_base_patch16_224,
    deit_tiny_patch16_224,
    deit_small_patch16_224,
    deit_base_patch16_224
)
from .sequential_pe import (
    SequentialPE, SinusoidalPositionalEmbedding, RoFormerSinusoidalPositionalEmbedding,
    RoFormer2DPositionalEmbedding, RoFormer2DMixedPositionalEmbedding
)


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_unk=config.TRAIN.USE_UNK_POS,
            use_idx_emb=config.TRAIN.USE_IDX_EMB,
            use_dlocr=config.TRAIN.USE_DLOCR,
            dlocr_type=config.TRAIN.DLOCR_TYPE,
            pe_type=config.PE_TYPE,
            pe_apply_method=config.PE_APPLY_METHOD,
            pe_dim=config.PE_OUT_PROJ_DIM,
            use_pe_multi_head=config.USE_PE_MULTI_HEAD,
            use_pe_qk_per_layer=config.USE_PE_QK_PER_LAYER,
        )
    elif model_type == 'resnet50':
        model = ResNet50(
            num_classes=config.MODEL.NUM_CLASSES
        )
    elif model_type == 'vit_b_16':
        model = vit_base_patch16_224(
            pretrained=False,
            num_classes=config.MODEL.NUM_CLASSES,
            use_unk=config.TRAIN.USE_UNK_POS,
            use_idx_emb=config.TRAIN.USE_IDX_EMB,
            use_dlocr=config.TRAIN.USE_DLOCR,
            dlocr_type=config.TRAIN.DLOCR_TYPE,
            attn_drop_rate=config.MODEL.ATTN_DROP_RATE,
            drop_rate=config.MODEL.DROP_RATE,
            pe_type=config.PE_TYPE,
            pe_apply_method=config.PE_APPLY_METHOD,
            pe_dim=config.PE_OUT_PROJ_DIM,
            use_pe_multi_head=config.USE_PE_MULTI_HEAD,
            use_pe_qk_per_layer=config.USE_PE_QK_PER_LAYER,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    elif model_type == "vit_s_16":
        model = vit_small_patch16_224(
            pretrained=False, 
            num_classes=config.MODEL.NUM_CLASSES,
            use_unk=config.TRAIN.USE_UNK_POS,
            use_idx_emb=config.TRAIN.USE_IDX_EMB,
            use_dlocr=config.TRAIN.USE_DLOCR,
            dlocr_type=config.TRAIN.DLOCR_TYPE,
            pe_type=config.PE_TYPE,
            pe_apply_method=config.PE_APPLY_METHOD,
            pe_dim=config.PE_OUT_PROJ_DIM,
            use_pe_multi_head=config.USE_PE_MULTI_HEAD,
            use_pe_qk_per_layer=config.USE_PE_QK_PER_LAYER,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    elif model_type == "deit_tiny_16":
        model = deit_tiny_patch16(
            pretrained=False, 
            num_classes=config.MODEL.NUM_CLASSES,
            use_unk=config.TRAIN.USE_UNK_POS,
            use_idx_emb=config.TRAIN.USE_IDX_EMB,
            use_dlocr=config.TRAIN.USE_DLOCR,
            dlocr_type=config.TRAIN.DLOCR_TYPE,
        )        
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model


def build_pe_model(config):
    pe_type = config.PE_TYPE
    pe_apply_method = config.PE_APPLY_METHOD

    if pe_type == "seq_pe":
        pe_model = SequentialPE(
            pe_embed_dim=config.PE_EMBED_DIM,
            num_attention_heads=config.NUM_ATTENTION_HEADS,
            layer_num=config.SEQPE_LAYER_NUM,
            max_digits=config.SEQPE_MAX_DIGITS,
            attn_direction=config.SEQPE_ATTN_DIRECTION,
            use_last_layernorm=config.SEQPE_LAST_LAYERNORM,
            mask_padding=config.SEQPE_MASK_PADDING,
            activation_function=config.SEQPE_ACTIVATION_FUNCTION,
            resid_pdrop=config.SEQPE_RESID_PDROP,
            attn_pdrop=config.SEQPE_ATTN_PDROP,
            scale_attn_weights=config.SEQPE_SCALE_ATTN_WEIGHTS,
            use_cls_token=True,
            out_proj_dim=config.PE_OUT_PROJ_DIM,
            seqpe_temperature=config.SEQPE_TEMPERATURE,
            seqpe_init_norm_weight=config.SEQPE_INIT_NORM_WEIGHT if hasattr(config, 'SEQPE_INIT_NORM_WEIGHT') else 1.0,
            data_dim=config.PE_DATA_DIM
        )
    elif pe_type == "sin":
        # +1 is for the cls token
        pe_model = SinusoidalPositionalEmbedding(
            num_positions=config.PE_MAX_POSITION + 1,
            embedding_dim=config.PE_OUT_PROJ_DIM,
            base=config.SINUSOIDAL_PE_BASE
        )
    elif pe_type == "rotary":
        # +1 is for the cls token
        if pe_apply_method == "rotary":
            pe_model = RoFormerSinusoidalPositionalEmbedding(
                num_positions=config.PE_MAX_POSITION + 1,
                embedding_dim=config.PE_OUT_PROJ_DIM,
                base=config.SINUSOIDAL_PE_BASE
            )
        elif pe_apply_method == "rotary_2d":
            pe_model = RoFormer2DPositionalEmbedding(
                num_positions=config.PE_MAX_POSITION,
                embedding_dim=config.PE_OUT_PROJ_DIM,
                base=config.SINUSOIDAL_PE_BASE                
            )
        elif pe_apply_method == "rotary_mixed":
            pe_model = RoFormer2DMixedPositionalEmbedding(
                num_positions=config.PE_MAX_POSITION,
                embedding_dim=config.PE_OUT_PROJ_DIM,
                num_heads=config.MODEL.NUM_HEADS,
                vit_depth=config.MODEL.DEPTH,
                base=config.SINUSOIDAL_PE_BASE
            )
        else:
            raise NotImplementedError
    elif pe_type in ["vanilla", "nope"]:
        pe_model = None
    elif pe_type == "alibi":
        raise NotImplementedError
    else:
        raise NotImplementedError
    return pe_model