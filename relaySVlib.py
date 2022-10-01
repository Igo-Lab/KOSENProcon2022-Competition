import tkinter as tk
import requests
import json
from tkinter.scrolledtext import ScrolledText
import main_GUI

#パラメータの定義
params={'procon-token':main_GUI.TOKEN}
P_ID = []
answers = []

#分割数送信
def POSTrequest_problem(how):
    ge=int(how.get())
    burl='https://procon33-practice.kosen.work/problem/chunks'
    keywad={'token':main_GUI.TOKEN,'n':''+str(ge)}
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
        curl.append('https://procon33-practice.kosen.work/problem/chunks'+str(num[i]))
        wsyu.append(requests.get(curl[i],params=params))
        print(num[i])
        i=i+1



#問題情報の取得
def syutoku(P_ID):
    surl='https://procon33-practice.kosen.work/problem'
    res=requests.get(surl,params)
    print(res.status_code)
    print(res.json())
    P_ID = json.loads(res.json())

#文字数制限
def limit_char(st_lim):
    return len(st_lim) <= 1

#回答を送信する
def POSTrequest_answer(P_ID,answers):
    hed = {'Content-Type':'application/json'}
    payload = {'problem_id':P_ID,'answers':answers}
    res = requests.post(url='https://procon33-practice.kosen.work/problem',headers=hed,params=params,data=payload)
    print(res.status_code)
    print(res.json())