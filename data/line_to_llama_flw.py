from os import listdir
import os
from os.path import isfile, join
import csv


# 自己的LINE帳號名稱
# master = "陳奕利Jefferson Chen"
# 放LINE聊天記錄txt檔的目錄
data_dir = "./"

# 輸出檔案的檔名
output_file_name = "training_data_flw.csv"


instructions_list = []
inputs_list = []
outputs_list = []

master_name = "陳奕利Jefferson Chen"


def isMaster(name):
    return master_name in name


# 讀每個檔案
def create_formatted_content(file_name):
    instruction = ""
    input = ""
    output = ""
    lines = None
    with open(file_name, encoding="utf-8") as f:
        lines = f.readlines()
    if lines == None:
        return
    pre_is_master = False

    for i in range(4, len(lines)):
        # 去掉空白行
        if (lines[i]) == "\n":
            continue
        if lines[i].endswith("已收回訊息"):
            continue
        w = lines[i].split("\t")
        # 去掉日期那行
        if len(w) < 3:
            continue
        if "收回訊息" in w[2]:
            continue
        if isMaster(w[1]):
            output += w[2]
            pre_is_master = True
        else:
            if pre_is_master:
                instructions_list.append(instruction)
                inputs_list.append(input)
                outputs_list.append(output)
                instruction = ""
                input = ""
                output = ""
            instruction += w[2]
            pre_is_master = False


# 輸出檔案
def output_file(instructions_list, inputs_list, outputs_list):
    # 三個list不一樣就不處理
    if (
        len(instructions_list) != len(inputs_list)
        or len(inputs_list) != len(outputs_list)
        or len(instructions_list) != len(outputs_list)
    ):
        return
    block_title = "=你是好友，也是大學同學。請以好友的回氣回答對話。"
    with open(output_file_name, "w", encoding="utf-8") as writer:
        fieldnames = ["instruction", "input", "output", "text"]
        writer = csv.DictWriter(writer, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(instructions_list)):
            writer.writerow(
                {
                    "instruction": instructions_list[i],
                    "input": "",
                    "output": outputs_list[i],
                    "text": block_title,
                }
            )


if __name__ == "__main__":
    # 拿到所有是聊天記錄的txt檔
    files = [
        f
        for f in listdir(data_dir)
        if isfile(join(data_dir, f)) and f.endswith("的聊天.txt")
    ]
    #
    for f in files:
        create_formatted_content(f)
    output_file(instructions_list, inputs_list, outputs_list)
    print("done...")
    print(f"total length: {len(instructions_list)}")
    file_size = os.path.getsize(output_file_name)
    print(f"data size: {file_size/1024**2}mb")
