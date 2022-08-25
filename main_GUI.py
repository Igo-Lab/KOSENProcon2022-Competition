import tkinter as tk
from tkinter import messagebox
import subprocess
import call_test
import numpy as np

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

    send_config(prb_data,sum_list)

def send_config(data,list):
    wid = tk.Tk()

    wid.geometry('500x250')
    wid.title('回答送信確認画面')

    #送信ボタン
    send_btn = tk.Button(wid,text='修正せずに回答を送信',command=post_ans)
    send_btn.place(x=50,y=50)


    #修正した回答の送信ボタン
    fixed_send_btn = tk.Button(wid,text='修正した内容を送信',command=post_fix_ans)
    fixed_send_btn.place(x=250,y=50)


def post_ans():
    print('自動判別の回答を送信した')

def post_fix_ans():
    print('修正済みの回答を送信した')


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