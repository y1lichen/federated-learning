from transformers import AutoTokenizer, TextStreamer
from hydra import compose, initialize
from server.utils.utils import set_parameters, get_init_parameters
from server.models import get_model

import flwr
import pandas as pd

# 特定user的對話記錄
chat_hist = "data/training_data_flw0.csv"
# 隨機取10筆聊天記錄放到instruction
chat_hist_df = pd.read_csv(chat_hist).sample(n=25)
# chat_hist_df = pd.read_csv(chat_hist).head(n=30)
with initialize(config_path="server/conf"):
    cfg = compose(config_name="config")


INSTRUCTION = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
# INSTRUCTION = "你是我的朋友，請你以朋友的口氣回答以下："
# INPUT = "明天要一起吃早餐嗎"
# INPUT = "健嗎"
# INPUT = "你微甲作業寫了？"
# INPUT = "你要去上統計學？"
INPUT = "我睡過頭了"

MODEL_NAME = cfg.model.name

# Load model and tokenizer
model = get_model(cfg.model)
parameters = get_init_parameters()
parameters = flwr.common.parameters_to_ndarrays(parameters)
set_parameters(model, parameters)
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

streamer = TextStreamer(tokenizer=tokenizer)
content = "你是好朋友，也是同學。請你以和過去相同的口氣回答問題，注意你的回答要符合對方的問題。"


templates = [{"role": "system", "content": content}]
for row in chat_hist_df.itertuples(index=True, name="Pandas"):
    templates.append({"role": "user", "content": row.instruction})
    templates.append({"role": "assistant", "content": row.output})
templates.append(
    {"role": "user", "content": INPUT},
)
# Generate answers
inputs = tokenizer.apply_chat_template(
    templates,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

outputs = model.generate(
    inputs,
    streamer=streamer,
    temperature=0.72,
    max_new_tokens=512,
    repetition_penalty=1.12,
    top_k=50,  # default 50
    top_p=0.95  # default 1.0
    # max_time=60.0,
)

print(">>> Ouput")
for i, sample_output in enumerate(outputs):
    print(
        "文案{}: {}".format(
            i + 1, tokenizer.decode(sample_output, skip_special_tokens=True)
        )
    )
