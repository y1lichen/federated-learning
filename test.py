from transformers import AutoTokenizer, TextStreamer
from hydra import compose, initialize
from server.utils.utils import set_parameters, get_init_parameters
from server.models import get_model
import flwr


with initialize(config_path="server/conf"):
    cfg = compose(config_name="config")


INSTRUCTION = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
# INSTRUCTION = "你是我的朋友，請你以朋友的口氣回答以下："
# INPUT = "明天要一起吃早餐嗎"
# INPUT = "健嗎"
# INPUT = "你微甲作業寫了？"
INPUT = "你要去上統計學？"
# INPUT = "我睡過頭了"

MODEL_NAME = cfg.model.name

# Load model and tokenizer
model = get_model(cfg.model)
parameters = get_init_parameters()
parameters = flwr.common.parameters_to_ndarrays(parameters)
set_parameters(model, parameters)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

streamer = TextStreamer(tokenizer=tokenizer)
# Generate answers
# system_prompt = f"[INST] {INSTRUCTION} [/INST]\n [USER] {INPUT} [/USER]"
# inputs = tokenizer(system_prompt, return_tensors="pt")
inputs = tokenizer.apply_chat_template(
    [
        {
            "role": "system",
            "content": "你是好朋友，同是也是同學",
        },
        {"role": "user", "content": INPUT},
    ],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

outputs = model.generate(
    inputs,
    streamer=streamer,
    temperature=0.62,
    max_new_tokens=1024,
    repetition_penalty=1.15,
    max_time=60.0,
)

print(">>> Ouput")
for i, sample_output in enumerate(outputs):
    print(
        "文案{}: {}".format(
            i + 1, tokenizer.decode(sample_output, skip_special_tokens=True)
        )
    )
