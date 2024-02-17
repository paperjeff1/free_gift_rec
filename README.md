# free_gift_rec
## 目的 : 使用MF方法推薦商品，以用戶購買商品紀錄作為評分矩陣
***
檔案說明:
1. 訓練區: 定期產生訓練模型，並上傳至MLflow
* train.py: 呼叫model.py模組，模式代train來建模
* model.py: if 模式是train則建模並上傳模型檔案、相關參數
***
2. 預測區: 透過All9funData.py建立預測資料集、預測資料，由技術組控制
* jjna_free_gift_build.py: 建立預測資料集，由All9fun呼叫，抓資料並整理產出csv結果，通常是id+特徵值的dataframe，回傳給All9fun
* model.py: All9fun呼叫，模式為test時，把預測資料集放進之前訓練時建立的模型，再做一些後處理，產出結果回傳給All9fun
***
3. 其他檔案:
* 破冰禮包對照.csv: 分類名稱對應gifts_id
* 分類 - 排序細項.csv: 購買商品的次數
* id_list.json: 控制清單
* 排除清單流程.docx: 說明被預測ID 7天內不再預測的方式
