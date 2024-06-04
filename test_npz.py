# This python file is adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import torch
from transformers import AutoTokenizer, TextStreamer
from hydra import compose, initialize
from server.utils.utils import set_parameters, get_init_parameters
from transformers import AutoModelForCausalLM
from server.models import get_model
import flwr


with initialize(config_path="server/conf"):
    cfg = compose(config_name="config")

INSTRUCTION = "你是我的朋友，請你以朋友的口氣回答以下："
INPUT = "你要去上統計學嗎"
MODEL_NAME = cfg.model.name

# Load model and tokenizer
model = get_model(cfg.model)
parameters = get_init_parameters()
parameters = flwr.common.parameters_to_ndarrays(parameters)
set_parameters(model, parameters)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

streamer = TextStreamer(tokenizer=tokenizer)
# Generate answers
inputs = tokenizer(f"{INSTRUCTION}" "{INPUT}", return_tensors="pt")
output = model.generate(
    **inputs,
    streamer=streamer,
    temperature=0.7,
    max_new_tokens=512,
    repetition_penalty=0.8,
    max_time=30.0,
)
output_text = tokenizer.decode(output[0], skip_special_tokens=False)

print(">>> Ouput")
print(output_text)
