import tkinter as tk
import requests
import json
from tkinter.scrolledtext import ScrolledText

#分割数送信
def bun():
    ge=int(how.get())
    burl='https://procon33-practice.kosen.work/problem/chunks'
    keywad={'token':'214756da1785484c9e578de68f9b58d421e8217816dc863cb8d28dd1274aac41','n':''+str(ge)}
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
        curl.append('https://procon33-practice.kosen.work/problem/chunks'+str(num[i]))
        tok={'token':'214756da1785484c9e578de68f9b58d421e8217816dc863cb8d28dd1274aac41'}
        wsyu.append(requests.get(curl[i],tok))
        print(num[i])
        i=i+1



#確認
def syutoku():
    surl='https://procon33-practice.kosen.work/problem'
    params={'token':'214756da1785484c9e578de68f9b58d421e8217816dc863cb8d28dd1274aac41'}
    res=requests.get(surl,params)
    nin.delete(0, tk.END)
    nin.insert(0,res.status_code)
    #print(res.json())
    #クエリパラメータ的な

#メインウィンドウ作成
root = tk.Tk()

#背景色
root.configure(bg="gold")

#サイズ
root.geometry("900x700")

#タイトル
root.title("かるたGUI")

#フレーム
frame=tk.Frame(root,relief=tk.SOLID,bd=1)
frame.grid()

#取得数送信ボタン
sousin = tk.Button(root, text="取得", command=bun,bg="OrangeRed",activebackground="Orange",fg="white",activeforeground="white",font=("HG正楷書体-PRO",17,"bold"))
sousin.place(x=53, y=50)

#確認ボタン
button = tk.Button(root, text="確認", command=syutoku,bg="OrangeRed",activebackgrou="Orange",fg="white",activeforeground="white",width=6,font=("HG正楷書体-PRO",17,"bold"))
button.place(x=530, y=50,width=60, height=41)

#処理開始ボタン
kai = tk.Button(root, text="START",bg="OrangeRed2",activebackground="Orange",fg="white",activeforeground="white",font=("HGP行書体",30, "bold"))
kai.place(x=300, y=270)

# 取得数ラベル
syulabel=tk.Label(master=root, text='取得数',font=("HGP行書体",20))
syulabel.place(x=120, y=17)

# 取得情報ラベル
syulabel=tk.Label(master=root, text='分割データファイル名', font=("HGP行書体",20,""))
syulabel.place(x=30, y=125)

#実行結果ラベル
kekka=tk.Label(master=root, text='実行結果', font=("HGP行書体",25))
kekka.place(x=30, y=350)

#取得数指定テキスト
how = tk.Entry(master=root, width=5, font=("",26),justify="center")
how.place(x=130, y=50,width=69, height=40)

#確認テキスト
nin=tk.Entry(master=root,width=3,font=("HGP行書体",35,""))
nin.place(x=600,y=50,width=70, height=41)

#分割データのファイル名取得のテキスト
text = ScrolledText(master=root, font=("", 14), height=5, width=70)
text.place(x=30,y=155)

#実行結果テキスト
vison = ScrolledText(master=root, font=("", 18), height=10, width=69)
vison.place(x=30,y=390)

root.mainloop()