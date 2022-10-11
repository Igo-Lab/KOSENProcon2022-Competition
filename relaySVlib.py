import tkinter as tk
from tkinter import messagebox
import requests
import json
from tkinter.scrolledtext import ScrolledText
import numpy as np

#ファイルの読み込み(試験的に使用。本番では先輩が書いてくれたスクリプトからimport)
try:
    with open("config.txt","r",encoding='utf-8') as CONFIG_FILE:
        TOKEN = CONFIG_FILE.readline()
        SV_URL = CONFIG_FILE.readline()
        CONFIG_FILE.close()

except FileNotFoundError:
    messagebox.showerror('エラー','config.txtが存在しません')

else:
    pass

#main_GUI.pyのC++呼び出しに渡す変数
problem = np.random.rand(5).astype(dtype=np.int16)
srcs = np.random.rand(5, 5).astype(dtype=np.int16)
len_problem = 5
lensrcs = np.full(5, 5, dtype=np.int32)

#### 以下、先輩が既にプログラム書いてくれているので不要 ####


#パラメータの定義
params={'procon-token':TOKEN}
P_ID = []
answers = []

#試合情報の取得
def GETrequest_match():
    res = requests.get(url=SV_URL+'match',params=params)
    code = res.status_code
    jsn = json.loads(res.json())
    return code,jsn

#問題情報の取得
def GETrequest_problem():
    surl=SV_URL+'problem'
    res=requests.get(surl,params)
    code = res.status_code
    print(res.json())
    info = json.loads(res.json())

    return code,info

#文字数制限
def limit_char(st_lim):
    return len(st_lim) <= 1

#取得する分割データの指定
def POSTrequest_problem(how):
    ge=int(how.get())
    burl=SV_URL+'problem/chunks'
    keywad={'token':TOKEN,'n':''+str(ge)}
    oto=requests.post(burl,params=keywad)
    print(oto.status_code)
    oput=oto.json() #分割データのファイル名が入っている
    print(oput)

    #wavの取得

    suu=ge
    #繰り返しのカウント変数
    i=0
    #jsonの要素を受け取るリスト
    num=[]
    #分割データのurlを受け取るリスト
    curl=[]
    #WAVファイルを受け取るリスト
    global wsyu
    wsyu=[]
    
    #wavを受け取るループ
    while  i<suu :
        num.append(oput['chunks'][i])
        curl.append(SV_URL+'problem/chunks'+str(num[i]))
        wsyu.append(requests.get(curl[i],params=params))
        print(num[i])
        i=i+1

#問題への回答
def POSTrequest_answer(P_ID,answers):
    hed = {'Content-Type':'application/json'}
    payload = {'problem_id':P_ID,'answers':answers}
    res = requests.post(url=SV_URL+'problem',headers=hed,params=params,data=payload)
    code = res.status_code
    print(res.json())
    return code
