import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from peft.utils import prepare_model_for_kbit_training
from peft import AutoPeftModelForCausalLM


import math


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def _get_lora_config(model_cfg: DictConfig):
    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )
    return peft_config


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
        )

    peft_config = _get_lora_config(model_cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        # quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(
        model,
        #  use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    return get_peft_model(model, peft_config)
