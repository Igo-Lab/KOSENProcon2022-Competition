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
    #繰り返しのカウント変数
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

#メインウィンドウ作成
root = tk.Tk()

#サイズ
root.geometry("900x600")

#タイトル
root.title("かるたGUI")

#フレーム
frame=tk.Frame(root,relief=tk.SOLID,bd=1)
frame.grid()

#取得数送信ボタン
sousin = tk.Button(root, text="データ取得", command=bun,bg="Light blue",activebackground="Orange")
sousin.place(x=180, y=50)

#確認ボタン
button = tk.Button(root, text="確認", command=syutoku,bg="Gray",activebackgrou="red",width=6,)
button.place(x=480, y=50)

#処理開始ボタン
kai = tk.Button(root, text="START", command=bun,bg="light green",activebackground="Orange",font=("weight",30))
kai.place(x=300, y=250)

# 取得数ラベル
syulabel=tk.Label(master=root, text='取得数', font=("",16))
syulabel.place(x=90, y=20)

# 取得情報ラベル
syulabel=tk.Label(master=root, text='分割データファイル名', font=("",16))
syulabel.place(x=30, y=100)

#実行結果ラベル
kekka=tk.Label(master=root, text='実行結果', font=("",16))
kekka.place(x=30, y=340)

#取得数指定テキスト
how = tk.Entry(master=root, width=5, font=("",20))
how.place(x=90, y=50)

#確認テキスト
nin=tk.Entry(master=root,width=3,font=("",15))
nin.place(x=540,y=50)

#分割データのファイル名取得のテキスト
text = ScrolledText(master=root, font=("", 12), height=5, width=90)
text.place(x=30,y=130)

#実行結果テキスト
vison = ScrolledText(master=root, font=("", 12), height=11, width=100)
vison.place(x=30,y=370)

root.mainloop()