from jj_free_gift_build import JJ_free_gift_dataset, JJ_free_gift
from model import data_modeling
import pandas as pd
import datetime
import os


'''
測試用
import importlib
import All9funData
importlib.reload(All9funData)
'''



class All9funData:
    # 建構式
    def __init__(self):
        pass


    # 產生預測用dataset
    def buildDataset(self,datasetAbsPath,buildDatasetParamsDic):
        # 欲回傳結果
        resultDict = {
            'result': False, # [False|True]，預設False，整個流程皆順利無錯通過最後才更改True
            'ext': None, # 想要額外給予的資訊，沒有則不需填入
        }

        # 流程控制
        flag = True

        # 產生需要預測的dataset
        if flag:
            try:
                #### path = os.path.join(os.getcwd(),str(datetime.datetime.now().date())+"_test"+".csv")
                test = JJ_free_gift_dataset(save=True, mode="test", path=datasetAbsPath)
                all_data = test.build()
                #如果data產出是空的
                if len(all_data) == 0:
                    raise Exception("data是空的")
            
            except Exception as e:
                flag = False
                print(e)
            
        #datasetAbsPath : path
        # 上述都順利則更改結果
        if flag:
            resultDict['result'] = True

        return resultDict

    
    # 透過MLflow平台模型進行預測(請依照實際需求改寫)
    def predict(self,modelPklAbsPath,datasetAbsPath,storeDirAbsPath):
        # 欲回傳結果
        resultDict = {
            'result': False, # [False|True]，預設False，整個流程皆順利無錯通過最後才更改True
            'ext': None, # 想要額外給予的資訊，沒有則不需填入
            'dictResult': None   #最後預測出來的結果
        }

        # 流程控制
        flag = True
       # modelPklAbsPath : current work place + model.pkl
    
        # MLflow平台模型進行預測，並回傳JSON格式的結果
        if flag:
            try:
                dictResult = data_modeling(mode="test",model_path=modelPklAbsPath,data=pd.read_csv(datasetAbsPath))
                resultDict['dictResult'] = dictResult
                #如果預測data是空的
                if (dictResult["result"] is None) or (len(dictResult["result"]) == 0):
                    raise Exception("predict列表是空的")
            except Exception as e:
                flag = False
                print(e)
            
        # 上述都順利則更改結果
        if flag:
            resultDict['result'] = True

        return resultDict

'''
#測試 buildDataset
datasetAbsPath = os.path.join(os.getcwd(),str(datetime.datetime.now().date())+"_test"+".csv")
buildDatasetParamsDic = 1
All9funData().buildDataset(datasetAbsPath, buildDatasetParamsDic)
'''

'''
#測試 predict
modelPklAbsPath = ??
datasetAbsPath = os.path.join(os.getcwd(),str(datetime.datetime.now().date())+"_test"+".csv")
storeDirAbsPath = ??
All9funData().predict(modelPklAbsPath, datasetAbsPath, storeDirAbsPath)
'''

