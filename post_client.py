import tkinter as tk
import requests
import json
from tkinter.scrolledtext import ScrolledText
import main_GUI

#分割数送信
def POSTrequest_problem(how):
    ge=int(how.get())
    burl='https://procon33-practice.kosen.work/problem/chunks'
    keywad={'token':main_GUI.TOKEN,'n':''+str(ge)}
    oto=requests.post(burl,params=keywad)
    print(oto.status_code)
    oput=oto.json()
    #text.delete('1.0', tk.END)
    print(oput)

    #wavの取得

    suu=int(ge)
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
        tok={'token':main_GUI.TOKEN}
        wsyu.append(requests.get(curl[i],tok))
        print(num[i])
        i=i+1



#ステータスコード確認
def syutoku():
    surl='https://procon33-practice.kosen.work/problem'
    params={'token':main_GUI.TOKEN}
    res=requests.get(surl,params)
    #nin.delete(0, tk.END)
    print(res.status_code)
    #クエリパラメータ的な

#文字数制限
def limit_char(st_lim):
    return len(st_lim) <= 1