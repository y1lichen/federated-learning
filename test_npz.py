# This python file is adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import torch
from transformers import AutoTokenizer
from fastchat.conversation import get_conv_template
from hydra import compose, initialize
from server.utils.utils import set_parameters, get_init_parameters
from transformers import AutoModelForCausalLM
from server.models import get_model
import flwr


with initialize(config_path="server/conf"):
    cfg = compose(config_name="config")

QUESTION = "你要去上統計學嗎"
MODEL_NAME = cfg.model.name
CONV_NAME = "twllm_v2"

# Load model and tokenizer
model = get_model(cfg.model)
parameters = get_init_parameters()
parameters = flwr.common.parameters_to_ndarrays(parameters)
set_parameters(model, parameters)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Generate answers
temperature = 0.7
choices = []
conv = get_conv_template(CONV_NAME)

conv.append_message(conv.roles[0], QUESTION)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer([prompt]).input_ids

output_ids = model.generate(
    input_ids=torch.as_tensor(input_ids),
    do_sample=True,
    temperature=temperature,
    max_new_tokens=1024,
)

output_ids = (
    output_ids[0]
    if model.config.is_encoder_decoder
    else output_ids[0][len(input_ids[0]) :]
)

# Be consistent with the template's stop_token_ids
if conv.stop_token_ids:
    stop_token_ids_index = [
        i for i, id in enumerate(output_ids) if id in conv.stop_token_ids
    ]
    if len(stop_token_ids_index) > 0:
        output_ids = output_ids[: stop_token_ids_index[0]]

output = tokenizer.decode(
    output_ids,
    spaces_between_special_tokens=False,
)

if conv.stop_str and output.find(conv.stop_str) > 0:
    output = output[: output.find(conv.stop_str)]

for special_token in tokenizer.special_tokens_map.values():
    if isinstance(special_token, list):
        for special_tok in special_token:
            output = output.replace(special_tok, "")
    else:
        output = output.replace(special_token, "")
output = output.strip()

conv.update_last_message(output)

print(f">> prompt: {prompt}")
print(f">> Generated: {output}")
