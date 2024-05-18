# 說明
程式碼是由[llm-flowertune](https://github.com/adap/flower/tree/main/examples/llm-flowertune)修改而來的。
修改的項目主要有：
1. 改用[Taiwan-LLM](https://huggingface.co/collections/yentinglin/taiwan-llm-6523f5a2d6ca498dc3810f07)
2. 調整訓練設定
3. 改用本地資料訓練
4. 拆分client & server

  在聯邦式學習中client和server勢必會分開部署，要降低client和server的藕合性。

# 操作說明

1. 準備訓練資料
    準備欄位為instruction, input, output, text的csv檔案。

2. 啟動server
```
python run server/run_server.py
```
3. 啟動client
    有幾個client要訓練就啟動幾個client

```
python client/run_client.py --i path/path_to_csv_data0.csv
python client/run_client.py --i path/path_to_csv_data1.csv
```
**目前server中的conf設定最多client上限為5個client，若有大於5個client需調整**

4. 測試訓練模型
    訓練peft檔設定儲存在根目錄下的results檔案夾中
```
python test.py --peft-path=/path/to/trained-model-dir/ \
    --question=“<問題內容>”
```
