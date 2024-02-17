# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:51:04 2023

@author: pc053
"""


from model import data_modeling
from datetime import datetime, timedelta, date
import yagmail

#import model
#import importlib
#importlib.reload(model)



def main():
    try:   #報錯時寄信
        data_modeling("train")
    except Exception as e:
    # 開始寄信
        start_time = datetime.now()
        receive = ['jeffyang@all9fun.com'] # 收信者
        sub = f"jj_免費玩家推薦 train程式錯誤訊息，執行錯誤時間為{start_time}" # 信件主旨
        content = f"錯誤訊息：\n 程式執行失敗\n {e} " # 信件內容
    
        yag = yagmail.SMTP(user = 'datasenderror@gmail.com', password = 'chjwhspbwwshzjtt') # 寄件者
        yag.send(to=receive, subject=sub, contents= content)
        
        #print(e)

#如果直接執行時，以下程式碼會被執行
if __name__ == '__main__':
    main()
