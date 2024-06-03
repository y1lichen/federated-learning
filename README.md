# 說明
程式碼是由[llm-flowertune](https://github.com/adap/flower/tree/main/examples/llm-flowertune)修改而來的。
修改的項目主要有：
1. 改用[Taiwan-LLM](https://huggingface.co/collections/yentinglin/taiwan-llm-6523f5a2d6ca498dc3810f07)
2. 調整訓練設定
3. 改用本地資料訓練

   重寫[custom dataset](https://github.com/y1lichen/federated-learning/blob/main/client/utils/custom_fds.py)。面對flower這種api迭代快速的新framework，programmer最高指導原則為不動可以運作的程式碼。看不懂custom_dfs的code很正常，因為我也不懂。
4. 拆分client & server     
  在聯邦式學習中client和server勢必會分開部署，要降低client和server的藕合性。

5. 設定為只需一個client即可進行federated learning

6. 改使用.npz儲存訓練結果 
    註：用peft model儲存方法見base-FL分支
    
7. 目前main分支取消使用GPU 
    有關GPU相關設定，見base-FL分支

# 操作說明

1. 準備訓練資料
    準備欄位為instruction, input, output, text的csv檔案。

2. 啟動server
```
python run server/run_server.py
```
3. 啟動client

```
python client/run_client.py --i path/path_to_csv_data0.csv
```
