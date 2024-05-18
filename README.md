# 說明
程式碼是由[llm-flowertune](https://github.com/adap/flower/tree/main/examples/llm-flowertune)修改而來的。
修改的項目主要有：
1. 改用[Taiwan-LLM](https://huggingface.co/collections/yentinglin/taiwan-llm-6523f5a2d6ca498dc3810f07)
2. 調整訓練設定
3. 改用本地資料訓練
4. 拆分client & server
在聯邦式學習中client和server勢必會分開部署，要降低client和server的藕合性。

# 操作說明

