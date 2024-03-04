# ## JJ 免費玩家推薦
# Jeff新增 : 
# 1. 已經推薦的7天內不再推薦
# 2. 每天抓24小時內資料
# 3. 修正時裝配對BUG
# 4. Json可以輸出中文檔案
# 5. 將輸出結果[A,A,B]轉為[A1,A2,B1]
# 6. py檔可以寄信
# 7. py檔可以記錄每天模型的最佳參數
# 
# * PM提供的 '分類 - 排序細項.csv' 要定期維運，免費商品仍然會更新，先暫訂3個月一次
# * 主要採用Matrix Factorization方法
# https://everdark.github.io/k9/notebooks/ml/matrix_factorization/matrix_factorization.nb.html#23_one-class_matrix_factorization  
# https://zhuanlan.zhihu.com/p/69662980  
# https://kiddie92.github.io/2019/06/10/Matrix-Factorization%E7%AE%80%E4%BB%8B/
# * 程式架構 : 
# 1. 以所有綁鑽消費數據建立模型
# 2. 抓取當日上線未曾付費玩家，根據過去綁鑽紀錄，預測可能的綁鑽購買商品，該預測商品為沒買過的
# 3. 綁鑽商品會再進行歸類，如培養-卡牌
# 4. 輸出每個ID的3項可能購買商品類別，輸出jason檔案
# 5. 後端原廠依據商品類別，去對應應該推薦的付費商品
# * 除了目前模型 : Alternating Least Squares, Bayesian Personalized Ranking, Approximate Alternating Least Squares也可以嘗試
# ***

import urllib
import json
import pandas as pd
import requests
from datetime import datetime, timedelta, date
from io import StringIO
from scipy.sparse import csr_matrix
#from sklearn.metrics import mean_absolute_percentage_error
from implicit.cpu.lmf import LogisticMatrixFactorization    #這套件下的其他演算法也可以研究，若要用GPU跑可改gpu
import implicit.evaluation
from collections import Counter
import warnings
import pickle as pkl
import mlflow
import mlflow.sklearn
import os
import shutil
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from decimal import Decimal
import sys
import socket
import random

#讀進商品分類表，後續匹配會須使用
#解決從外部呼叫時Python路徑的問題，先找到這份檔案的上一層資料夾，再合併路徑
p = os.path.join(os.path.dirname(__file__), "分類 - 排序細項.csv")
category = pd.read_csv(p)
category = category.set_index('品項')['分類-細分'].to_dict()  #索引為key，對應分類-細分欄位

p = os.path.join(os.path.dirname(__file__), "破冰禮包對照.csv")

#遊戲token
token = "Kv25TlPRTj1GZ1CgN2PswmQMNFKUH5118E9H152COToh09OLBsALS5QI1m8M1cv5"


#MLflow設定
#先獲得主機位置，要判斷是正式機還是測試機
def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
    s.close()
    return ip_address

ip_address = get_ip_address()

if ip_address == '210.242.121.36':         #正式機
    trackingServerUri = "http://210.242.121.36:8008"
elif ip_address == '210.242.121.32':       #測試機  
    trackingServerUri = "http://210.242.121.32:8008"
else:                                       #都不是就測試機，本機測試也是
    trackingServerUri = "http://210.242.121.32:8008"


os.environ["MLFLOW_TRACKING_USERNAME"] = "dadai"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "all9fun_dadai"
# 設置tracking server uri
mlflow.set_tracking_uri(trackingServerUri)
mlflow.set_experiment("jj_free_gift")
warnings.filterwarnings("ignore")

isRegisterModel = True # False不註冊，反之
registered_model_name = "jj_free_gift" # isRegisterModel=True才有效

#控制id清單的json檔案存放絕對路徑
#判斷在哪個作業系統執行
if sys.platform.startswith('linux'):
    folder_path = '/opt/services/AI_Project/gift/jj_free_gift'
else:
    folder_path = os.path.dirname(__file__)

file_name = 'id_list.json'
file_path = os.path.join(folder_path, file_name)



def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)


def data_modeling(mode,**kwargs):

    if mode == "train":

        #以下是要訓練的模式
        def clothes(x):
            '''由於時裝太多名稱，故將名稱對應回大的分類，後續建模採用這分類做為名稱'''
            if '购买服装' in x:
                return category[x]
            else:
                return x
    
        def find_count(x):
            '''抓出商品數量'''
            if "*" not in x:           #item_gain有些是空值
                return 1
            else:                      #有值的，抓取*後面的數量
                return int(x.split('*')[1])
    
        def item(x):
            '''將途徑、名稱連接在一起'''
            if x[1]=='':
                return x[0]
            else:
                return x[0]+'_'+x[1]
             
        SQL ='''
        select "#account_id",reason,item_gain
        from ta.v_event_23
        where ("$part_date" >='2020-08-01') AND ("$part_event" IN ('diamond_consume')) and diamond_consume_amount=0
        '''
        r = requests.post(url = 'http://210.242.105.89:8992/querySql?token='+token,  
                              headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                              data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        data = pd.read_csv(StringIO(s))
        data['item_gain'].fillna('',inplace = True)
        data['item_gain'] = data['item_gain'].apply(lambda x:str(x).split("\t")) #將多商品資料用"\t"分割，傳回list
        data = data.explode("item_gain") #將list解開巢狀結構，變成列
        data['count'] = data['item_gain'].apply(lambda x : find_count(x))   #取出商品數量
        data['item_gain'] = data['item_gain'].apply(lambda x:x.split('*')[0])  #取出商品名稱
        data['item'] = data.loc[:, ['reason',"item_gain"]].apply(lambda x:item(x),axis=1)  #連結途徑+商品名稱，代表1件商品
        data = data[data['item'].isin(list(category.keys()))]    #限定商品只能在'分類 - 排序細項.csv'中，其他的難以推薦
        # data['map_cat'] = data['item'].map(category)             # 創建對應類別，用於篩選
        # data = data[~data['map_cat'].str.contains('培養')]        # 排除培養類別
        data['item'] = data['item'].apply(lambda x:clothes(x))   #把時裝名稱替換，改為時裝大的歸類 (來自excel)

        data = pd.crosstab(data['#account_id'],data['item'],values=data['count'],aggfunc=sum) #將直的內容展開
        data = data.reset_index()
        data.fillna(0,inplace=True)
        
        #將完整的購買清單表寫入excel儲存，未來預測清單需要使用
        data.to_csv(folder_path + '/all_buy_list.csv', index = False, mode = 'w', encoding = 'utf_8_sig')
        
        #繼續將資料處理為模型需求
        user = data["#account_id"]
        user_map = dict()
        reverse_user_map = dict()
        for i in range(len(user)):
            user_map[i] = user[i]
            reverse_user_map[user[i]] = i      #紀錄每個ID對應的索引編號
        item = data.columns[1:]
        item_map = dict()
        for i in range(len(item)):
            item_map[i] = item[i]
    
        data = data.set_index("#account_id")
        
        # 將資料用比較有效率的方式儲存起來
        # 建立MinMaxScaler標準化物件
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.values)
        user_item_data = csr_matrix(data_scaled)   #因為資料很多0，稀疏矩陣可以優化的儲存模式
        data = data.reset_index()
        data = pd.melt(data, id_vars=['#account_id'], value_vars=data.columns[1:])   #將cross_table收回為直式
        data = data[data['value']!=0]
        del data_scaled
        
        #訓練模型
        train, test = implicit.evaluation.train_test_split(user_item_data)
        
        
        # 跑迴圈測試最佳參數
        result = dict()
        for f in [10,30,50,70,90,150]:
            for reg in [1e-1,1e-2,1,1e1,1e2,1e3]:
                for learning_rate in [1,10,1e-1,1e-2]:
                    for iterations in [30,50,100]:
                        model = LogisticMatrixFactorization(factors=f, regularization=reg,iterations=iterations, learning_rate=learning_rate)
                        model.fit(train)
                        ans = implicit.evaluation.precision_at_k(model, train, test, K=3, show_progress=False)   
                        #推3項商品，k是隱類別，有時難以直觀解釋代表哪項商品
                        # Precision at k is the proportion of recommended items in the top-k set that are relevant
                        # print((f,reg,learning_rate,iterations),ans)
                        result[(f,reg,learning_rate,iterations)] = ans
        
        
        
        #採用最佳precision的參數放進模型，precision每人都有一個值，最後算下來平均
        result = {k: v for k, v in sorted(result.items(), key=lambda item: -item[1])}  #peter寫法
        max_key = max(result, key = result.get)  #取出最大值的鍵   若有多個也只會回傳一個   GPT寫法，感覺簡單一點
        f, reg, learning_rate, iterations = max_key
        #f,reg,learning_rate,iterations = 70 ,100.0, 1, 50 #list(result)[0]
        
        #print(f,reg,learning_rate,iterations, result[max(result, key = result.get)])   #呈現參數 + precision
        #print(round(result[(f,reg,learning_rate,iterations)],3))
        model = LogisticMatrixFactorization(factors=f, regularization=reg, learning_rate=learning_rate,iterations=iterations)
        model.fit(user_item_data)
        
        
        model_name = "jj_free_gift"
   #    check = True
   #    while check: 
        
   
        with mlflow.start_run():
 
            #   automl.fit(data.drop(["lost","date"],axis=1), data["lost"], **settings,eval_method="auto", n_jobs=-1,verbose=0) 
            #   if recall_score(data["lost"],automl.predict(data.drop(["lost","date"],axis=1)))!=0 :
            #       check = False     
            # MLflow寫入parameters   
            mlflow.log_param("f", f)
            mlflow.log_param("reg", reg)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("iterations", iterations)
                
            # MLflow寫入metrics
            mlflow.log_metric("precision", result[max_key])
            #mlflow.log_metric("precision", 0.35)  
        
        
            # MLflow上傳model
            if isRegisterModel==False:
                mlflow.sklearn.log_model(model, "model", serialization_format='pickle')
            else:
                mlflow.sklearn.log_model(model, "model", registered_model_name=registered_model_name, serialization_format='pickle')
            
            #這段是存在本機端才需要的
            #用mlflow套件，把模型儲存起來成一個檔案，在資料夾temp下，存為PICKLE format              
            #mlflow.sklearn.save_model(model, path, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
            
            #不存在會自己創建
            #with open('automl.pkl', 'wb') as f:
            #    pkl.dump(model, f, pkl.HIGHEST_PROTOCOL)      
            
            
            path = os.path.join(os.getcwd(), "temp")
            #創建temp資料夾
            os.mkdir(path)
            #temp下創建all9fun_package資料夾
            os.mkdir(os.path.join(path,"all9fun_package"))

            
             # 將上傳資料複製到all9fun_package資料夾
            for file in ["model.py","All9funData.py","jj_free_gift_build.py","train.py","test.csv","分類 - 排序細項.csv","破冰禮包對照.csv"]:
                if os.path.exists(os.path.join(os.getcwd(),file)):
                    shutil.copy(os.path.join(os.getcwd(),file), os.path.join(path,"all9fun_package",file))
            # data["all9fun_id"] = all9funid 
            # data.set_index("all9fun_id").to_csv(os.path.join(path,str(datetime.datetime.now().date())+"_train"+".csv"))
            
            
            #把temp資料的檔案，包括all9fun_package資料夾丟到 ./mount/mlflow_store
            mlflow.log_artifacts(path)
             

        # 抓取MLflow上的版本資訊
        client = MlflowClient()
        model_version_infos = client.search_model_versions("name = '%s'" % model_name)
        

        
        # 找出最新版本，跑wait_until_ready
        new_model_version = max([int(model_version_info.version) for model_version_info in model_version_infos])
        wait_until_ready(model_name, new_model_version)
        
        # Update最新版本
        client.update_model_version(
        name=model_name,
        version=new_model_version,
        description="This model version is new version."
        )
        
        # 把最新版本的模型的階段改成Production
        client.transition_model_version_stage(
        name=model_name,
        version=new_model_version,
        stage="Production"
        )
        
        mlflow.end_run()
        
        #最後刪除temp資料夾
        if os.path.exists(path):
            shutil.rmtree(path)
        #with open('model.pkl', 'wb') as files:
        #     pkl.dump(automl, files)
        
        
    else:
    # mode為test時

        predict = kwargs["data"]   #把建立好的dataset讀出來
        #抓取當日DAU (排除付費玩家)，後續建立預測清單用
        #因為之前建立dataset時就已經篩過一次，不用重撈SQL了

        # 讀入建立模型使用的全部購買data
        data = pd.read_csv(folder_path + '/all_buy_list.csv')

        #建立用戶及商品矩陣
        user = data["#account_id"]
        user_map = dict()
        reverse_user_map = dict()
        for i in range(len(user)):
            user_map[i] = user[i]
            reverse_user_map[user[i]] = i      #紀錄每個ID對應的索引編號
        item = data.columns[1:]
        item_map = dict()
        for i in range(len(item)):
            item_map[i] = item[i]
        
        data = data.set_index("#account_id")
        
        # 將資料用比較有效率的方式儲存起來
        # 建立MinMaxScaler標準化物件
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.values)
        user_item_data = csr_matrix(data_scaled)   #因為資料很多0，稀疏矩陣可以優化的儲存模式
        data = data.reset_index()
        data = pd.melt(data, id_vars=['#account_id'], value_vars=data.columns[1:])   #將cross_table收回為直式
        data = data[data['value']!=0]
        del data_scaled

        
        try:
                model_path = kwargs["model_path"]
                with open(model_path, 'rb') as f:
                    model = pkl.load(f)
                    
                # 先推前50項商品，再把培養類別刪除
                def item_predict(x):
                    try:
                        userid = reverse_user_map[x]     #回傳ID對應的索引編號，這編號來代表userid
                        ids, scores = model.recommend(userid, user_item_data[userid], N=50         #id, id對應的向量，推薦3項商品
                                              , filter_already_liked_items=True)   #不推薦已經購買的商品，輸出3個商品序列、3個商品分數
                        rec_item = [category.get(item_map[x], item_map[x]) for x in ids]   #商品序列對應到商品名稱，再用名稱對應3類別的list
                        # 這邊使用dict.get是為了時裝因為做過轉換了，所以不會對應到對照表，會以原值表示 (已轉換過)
                        #  cat = most_frequent(rec_item)
                        rec_raw = [item_map[x] for x in ids]    # 推薦原始商品

                        # 以下為培養類別處理
                        # rec_item = ['培養球員', '時裝', '培養技能', '培養技能', '培養技能', '慶祝']
                        # rec_raw = ['3', '1', '2', '3', '1', '1']
                        # scores = ['a', 'dddd', 'ff', '3', '1', '9']
                        filter_index = [index for index, item in enumerate(rec_item) if '培養' not in item]         # 排除培養
                        filter_index = filter_index[:3]                                                            # 取前3
                        rec_item, rec_raw, scores = map(list, zip(*[(rec_item[x], rec_raw[x], scores[x]) for x in filter_index]))    # 只取培養的位置元素，轉為列表

                        # 如果排除培養後，推薦商品元素不滿3個
                        if len(rec_item) < 3:
                            all_cat = list(set(list(category.values())))      # 把分類字典的值，取出唯一，再排除培養，這是隨機抽樣補商品清單
                            all_cat = [item for item in all_cat if '培養' not in item]
                            add_sample_list = random.sample(all_cat, 3 - len(rec_item))                  # 看缺多少補為3個商品

                            rec_item.extend(add_sample_list)                                 # 補商品
                            rec_raw.extend(['隨機補商品']* (3 - len(rec_raw)))                 # 補商品細項 : 隨機補商品
                            scores.extend([0] * (3 - len(scores)))                           # 補分數為0

                        return rec_item, rec_raw, scores  #cat
                    except:                               #沒綁鑽消費紀錄，故不推薦，或消費的商品被排掉了
                        return 'no data','no data','no data'

                # predict = pd.read_csv('dataset.csv')

                #回傳推薦類別、細項、預測分數
                #predict['rec'] = predict['#account_id'].apply(lambda x:item_predict(x))
                predict[['rec', 'ids', 'scores']] = predict['#account_id'].apply(item_predict).apply(pd.Series)
                predict = predict[predict['rec']!='no data']     # 把沒資料的排除掉，3個欄位都是no data
                
                #每列取3項物品的平均分數
                predict['avg_scores'] = predict['scores'].apply(pd.Series).mean(axis=1)
                #依據平均預測分數分組，低中高
                predict['pred_proba'] = pd.qcut(predict['avg_scores'], 3, labels=['low','medium', 'high'])
                
                #print(predict['pred_proba'].value_counts())
                #print(predict[['scores','avg_scores', 'pred_proba']].sort_values(by='avg_scores'))
                # 定義一個函數來將浮點數轉換成 Decimal 物件，round精度不夠取完後至Json仍會有多位數
                def float_to_decimal(value):
                    return float(Decimal(str(value)).quantize(Decimal('0.00')))
                
                predict['avg_scores'] = predict['avg_scores'].apply(float_to_decimal)
                
                
                is_group = 0   # 0代表不分組
                if is_group == 1:
                    #分AB組
                    A, B = train_test_split(predict, test_size=0.5, random_state=0, stratify = predict["pred_proba"])
                    A['group'] = 'A'
                    B['group'] = 'B'
                    predict = pd.concat([A,B],axis=0)
                elif is_group == 0:
                    #不分組，全為A組
                    predict['group'] = 'A'    
                
                     
                #只篩選需要的欄位
                predict = predict[['#account_id','rec','avg_scores','pred_proba','group']]
                      
                
                #本段將原資料型態[A,A,B]，轉成[A1,A2,B1]，方便原廠匹配禮包
                predict['count'] = predict['rec'].apply(lambda x : Counter(x))   #統計每列的品項有幾個，型態為字典，如{'培養-技能': 2, '球員': 1}
                predict['count'] = predict['count'].apply(lambda x : [str(k)+str(v) for k, v in x.items()])  #將字典轉為列表
                
                for i in predict.index:
                    if len(predict.loc[i,'count']) == 3:   #如果共3項就不用處理
                        continue
                    elif len(predict.loc[i,'count']) == 2:
                        key = [string for string in predict.loc[i,'count'] if string[-1] == '2']  #將列表中尾數為2的字串找出，結果為列表
                        predict.loc[i,'count'].insert(0, key[0][:-1]+str('1')) #將2結尾的，增加一個1結尾的字串放進列表
                    elif len(predict.loc[i,'count']) == 1:
                        predict.loc[i,'count'].insert(0, predict.loc[i,'count'][0][:-1]+str('2'))
                        predict.loc[i,'count'].insert(0, predict.loc[i,'count'][0][:-1]+str('1'))
                
                predict['rec'] = predict['count']
                predict.drop(columns=['count'], inplace = True)
                
                #後面要做id控制清單用的，這邊先不使用
                result = predict.set_index('#account_id')['rec'].to_dict()
                
                
                #將禮包轉成gifts_id
                # 定義函數，將 rec 中的元素轉換為對應的gifts_id
                gift_map_df = pd.read_csv(p, dtype={'gifts_id': str})   
                def map_rec_to_num(rec):
                    return [gift_map_df.loc[gift_map_df['分類-細分'] == r, 'gifts_id'].iloc[0] for r in rec]
                
                '''
                #預防沒匹配時，給一個預設值
                def map_rec_to_num(rec, default_val='empty'):
                    num_list = []
                    for r in rec:
                        result = gift_map_df.loc[gift_map_df['分類-細分'] == r, 'gifts_id']
                        if not result.empty:
                            num_list.append(result.iloc[0])
                        else:
                            num_list.append(default_val)
                    return num_list
                '''
                
                # 將 rec 中的元素轉換為對應的編號
                predict['rec'] = predict['rec'].apply(lambda x: map_rec_to_num(x))


                
                #將dataframe轉成輸出格式, [{"ID1":"55688", "group":"A"}, ...]
                predict = predict.rename(columns={'rec': 'gifts_id'})
                predict['#account_id'] = predict['#account_id'].astype(str)
                
                #print(predict)
                
                
                flag = False
                #測試特定ID派發獎勵用，False表示不測試
                if flag:
                    id_test_list = ['204621','46896','1564370','1374041','1312388','abcd1234aa']   #要測試的id
                    group_test_list = ['A','A','A','B','B','B']      #要測試的組別
                    predict = predict.iloc[:len(id_test_list), :] #只保留對應ID數量的列
                    predict['#account_id'] = id_test_list   #把原ID替換為要測試的ID
                    predict['group'] = group_test_list
                
                
                
                result_list = []
                #逐列讀取
                for index, row in predict.iterrows():
                    row_dict = {}
                    
                    #將每個欄位的資料轉為字典
                    for key, value in row.items():
                        row_dict[key] = value

                    result_list.append(row_dict)

                #print(result_list)
         
                ### report   產出串接需求的文檔格式
                report = dict()
                report["result"] = result_list
                report["#event_name"] = "JJ_free_gift"
                report["#type"]="track"
                report["ts"] = int(str(time.time()).split(".")[0]) 

            
        except Exception as e:
                print(e)
                print("model not found")

        
        #另外更新預測ID清單，用Json做控制，7天內領過的不再發放
        #先讀入舊的清單，再將今日預測清單值設定空值，更新，最後存入json
        try:
            
            result = {k : '' for k, v in result.items()}   # id:領到預測日
            
            try:  #若檔案已經存在
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                    data = {int(k): v for k, v in data.items()}     #把Json讀出來的鍵值轉為int
    
                data.update(result)
    
                with open(file_path, 'w') as f:
                    json.dump(data, f)
            except:   #若檔案不存在
                with open(file_path, 'w') as f:
                    json.dump(result, f)
            
            return report
            
        except Exception as e:
            print(e)
            print('result未成功建立')
