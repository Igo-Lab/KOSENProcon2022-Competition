import tkinter as tk
import requests
import json
from tkinter.scrolledtext import ScrolledText

#分割数送信
def bun():
    ge=int(how.get())
    burl='url'
    keywad={'token':'トークン','n':''+str(ge)}
    oto=requests.post(burl,params=keywad)
    print(oto.status_code)
    oput=oto.json()
    text.delete('1.0', tk.END)
    text.insert('1.0',oput)

    #wavの取得
    suu=int(ge)
    i=0
    num=[]
    curl=[]
    global wsyu
    wsyu=[]
    while  i<suu :
        num.append(oput['chunks'][i])
        curl.append('url'+str(num[i]))
        tok={'token':'トークン'}
        wsyu.append(requests.get(curl[i],tok))
        print(num[i])
        i=i+1



#確認
def syutoku():
    surl='url'
    params={'token':'トークン'}
    res=requests.get(surl,params)
    nin.delete(0, tk.END)
    nin.insert(0,res.status_code)
    #print(res.json())
    #クエリパラメータ的な

root = tk.Tk()

root.geometry("1000x300")

root.title("かるたGUI")

#取得数送信ボタン
sousin = tk.Button(root, text="取得数送信", command=bun)
sousin.place(x=180, y=50)

#確認ボタン
button = tk.Button(root, text="確認", command=syutoku)
button.place(x=500, y=50)

# 取得数ラベル
syulabel=tk.Label(master=root, text='取得数', font=("",16))
syulabel.place(x=90, y=20)

# 取得情報
syulabel=tk.Label(master=root, text='分割データファイル名', font=("",16))
syulabel.place(x=30, y=100)

#取得数指定テキスト
how = tk.Entry(master=root, width=5, font=("",20))
how.place(x=90, y=50)

#確認テキスト
nin=tk.Entry(master=root,width=3,font=("",15))
nin.place(x=540,y=50)

#分割データのファイル名取得のラベル
text = ScrolledText(master=root, font=("", 12), height=5, width=110)
text.place(x=30,y=130)

root.mainloop()