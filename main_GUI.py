import tkinter as tk
from tkinter import messagebox
import subprocess
import tkinter
from tokenize import String
import call_test
import numpy as np
from functools import partial
import requests

window_title = '音声解析の鬼 (GUIテスト)'

#分割データを取得
def get_prob_data():
    print('問題データを取得')
    #処理
    print('完了')

#解析ロジックの呼び出し
def call_main_prog():
    print('解析を開始')
    prb_data , sum_list = call_test.test()
    print('完了')

    send_ans(prb_data,sum_list)

#回答送信の確認画面
def send_ans(data,list):
    wid = tk.Tk()

    wid.geometry('600x250')
    wid.title('回答送信確認画面')

    #判別結果の画面出力と、修正画面
    #自動判別結果側
    box_name1 = tk.Label(wid,text='判別結果')
    box_name1.place(x=50,y=100)
    box_ans = tk.Entry(wid,width=50)
    box_ans.place(x=120,y=100)
    box_ans.insert(tk.END,str(data))
    #修正入力側
    box_name2 = tk.Label(wid,text='修正を入力')
    box_name2.place(x=50,y=120)
    box_fix = tk.Entry(wid,width=50)
    box_fix.place(x=120,y=120)

    #送信ボタン
    send_btn = tk.Button(wid,text='修正せずに回答を送信',command=partial(post_ans,data))
    send_btn.place(x=50,y=50)


    #修正した回答の送信ボタン
    fixed_send_btn = tk.Button(wid,text='修正した内容を送信',command=partial(post_fix_ans,box_fix))
    fixed_send_btn.place(x=250,y=50)
    



def post_ans(answer):
    print('自動判別の回答を送信した')
    print(answer)

def post_fix_ans(func):
    
    print('修正済みの回答を送信した')
    print(func.get())


frm = tk.Tk()

frm.geometry('1000x500')
frm.title(window_title)

#データ取得ボタンの定義と配置
get_prob_data_btn = tk.Button(frm,text='問題を取得',command=get_prob_data)
get_prob_data_btn.place(x=50,y=450)

#解析開始ボタンの定義と配置
start_main_program_btn = tk.Button(frm,text='解析開始!!',command=call_main_prog)
start_main_program_btn.place(x=150,y=450)

frm.mainloop()