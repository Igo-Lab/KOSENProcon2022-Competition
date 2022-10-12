from fileinput import close
from genericpath import isfile
from re import I
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tokenize import String, Token
import numpy as np
import numpy.typing as npt
from functools import partial
import os
from ctypes import *
import numpy as np

import call_test #仮でimportしている
import relaySVlib

from tkinter.scrolledtext import ScrolledText

TOKEN = relaySVlib.TOKEN
SV_URL = relaySVlib.SV_URL
PROBLEM = relaySVlib.problem
srcs = relaySVlib.srcs
len_problem = relaySVlib.len_problem
lensrcs = relaySVlib.lensrcs

def run_Logic(problem: npt.NDArray[np.int16], srcs: npt.NDArray[np.int16], len_problem, lensrcs: npt.NDArray[np.int32]):

    print(problem)
    print(srcs)
    print(len_problem)
    print(lensrcs)

    ### C++の呼び出し準備 ###

    cpp_resolver =cdll.LoadLibrary('./CPP_LOGIC.so')

    #C++関数の引数型
    _INT16_PP = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags="C")
    _INT16_P = POINTER(c_int16)
    _INT32_PP = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags="C")
    _INT32_P = POINTER(c_int32)

    #メインロジック関数インスタンス
    logic_resolver = cpp_resolver.resolver

    #C++関数の引数・戻り値定義
    logic_resolver.restype = None
    logic_resolver.argtypes = (_INT16_P, _INT16_PP, c_int32, _INT32_P, _INT32_PP)

    #srcのアドレスの二次元配列作成・resultのアドレスの二次元配列作成
    srcs_PP = (srcs.__array_interface__["data"][0] + np.arange(srcs.shape[0]) * srcs.strides[0]).astype(np.uintp)
    results = np.zeros((5, 2), dtype=np.int32)
    results_PP = (results.__array_interface__["data"][0] + np.arange(results.shape[0]) * results.strides[0]).astype(np.uintp)


    #メインロジックに丸投げ
    logic_resolver(problem.ctypes.data_as(_INT16_P),srcs_PP,len_problem,lensrcs,results_PP)

if __name__ == '__main__':

    window_title = '春光台にそそりたつ！！'

    #回答をなおす画面
    def retype_answer_frm():
        win = tk.Tk()

        win.geometry('700x350')
        win.title('回答を修正')
        win.configure(bg="black")

        #修正入力欄
        box_name2 = tk.Label(win,font=("normal",20),text='修正を入力',bg="black",fg="#ffffff")
        box_name2.place(x=0,y=10)
        box_fix = ScrolledText(win,font=("normal",16),height=5,width=50)
        box_fix.place(x=40,y=60)

        #送信ボタン
        send = tk.Button(win,background='#bbff99',text='修正を送信',font=("normal",20,"bold"),command=partial(send_ans_fixed,box_fix))
        send.place(x=50,y=250)

        #送信ボタン(修正なし)
        #send = tk.Button(win,background='#ff4444',text='修正せずに送信',font=("normal",20,"bold"))
        #send.place(x=450,y=250)

    #けっかはっぴょー
    def result_viewer(data,list):
        wid = tk.Tk()

        wid.geometry('700x520')
        wid.title('回答送信確認画面')
        wid.configure(bg="black")

        #判別結果の画面出力と、修正画面
        #自動判別結果側
        box_name1 = tk.Label(wid,font=("normal",20),text='判別結果',bg="black",fg="#ffffff")
        box_name1.place(x=0,y=0)
        box_ans = ScrolledText(wid,font=("normal",16),height=5,width=50)
        box_ans.place(x=40,y=60)
        box_ans.insert(tk.END,str(data))

        #差分合計値リストの表示
        list_name1 = tk.Label(wid,font=("normal",20),text='差分合計値リスト',bg='black',fg='#ffffff')
        list_name1.place(x=0,y=230)
        sum_list = ScrolledText(wid,font=("normal",16),height=4,width=50)
        sum_list.place(x=40,y=275)
        sum_list.insert(tk.END,str(list))

        #送信ボタン
        send_btn = tk.Button(wid,font=("normal",20,"bold"),text='修正せずに回答を送信',bg="#ff4444",command=partial(send_ans_default,box_ans))
        send_btn.place(x=350,y=420)


        #修正したいってときのボタン
        fixed_send_btn = tk.Button(wid,font=("normal",20,"bold"),text='回答を修正',bg="#bbff99",command=retype_answer_frm)
        fixed_send_btn.place(x=50,y=420)
        
        #画面下の注意書き
        attention_label = tk.Label(wid,font=("normal",10,"bold"),text='確認ダイアログは出ません!!',bg='red',fg='#ffffff')
        attention_label.place(x=360,y=480)

    #メイン画面に取得したファイル名を表示
    def print_filenames(tko,names):
        tko.insert(tk.END,names)

    #分割データを取得
    def get_prob_data(hmd):
        relaySVlib.GETrequest_problem()
        print('問題データを取得')
        print('分割データ数:'+ hmd.get())
        #処理
        print('完了')

    #パス出力の初期化
    def clear_path_list():
        got_file_name_list1.delete(0,tk.END)
        got_file_name_list2.delete(0,tk.END)
        got_file_name_list3.delete(0,tk.END)
        got_file_name_list4.delete(0,tk.END)
        got_file_name_list5.delete(0,tk.END)

    #ローカルファイルからの参照
    def get_prob_data_local():
        print("ローカルフォルダからデータを参照")
            #何らかの理由でpathlistが存在した場合に削除
        if os.path.isfile('pathlist') == True:
            os.remove('pathlist')
        else:
            pass

        clear_path_list()
        typ = [('All Files','*')]
        defdir = './procon22-wav/'
        if os.path.exists(defdir) == False:
            os.makedirs('./procon22-wav/')
        else:
            pass
        path = list(filedialog.askopenfilenames(filetypes=typ,initialdir= defdir))
            #ファイル読み込み後、要素数が不足している場合にNULLで補完する
        if len(path)<5:
            for i in range(5-len(path)):
                path.append('NULL')
        else:
            pass
        print_filenames(got_file_name_list1,os.path.basename(path[0]))
        print_filenames(got_file_name_list2,os.path.basename(path[1]))
        print_filenames(got_file_name_list3,os.path.basename(path[2]))
        print_filenames(got_file_name_list4,os.path.basename(path[3]))
        print_filenames(got_file_name_list5,os.path.basename(path[4]))
            #tkinterのボタンコマンドにおいて、うまく変数間で代入できないので外部ファイルにメモ
        f = open('pathlist','a',encoding='utf-8')

        for j in range(len(path)):
            f.write(path[j]+'\n')

        f.close()
        print("完了")

    #解析ロジックの呼び出し
    def call_main_prog():
        print('解析を開始')

        #resultofLogic = run_Logic(PROBLEM, srcs, len_problem, lensrcs)
        print('完了')
        #print(resultofLogic)
        result_viewer('test','test')

    #回答送信(修正加えたやつを送る)
    def send_ans_fixed(textbox):
        send_data = textbox.get("1.0", "end-1c")
        print(send_data)

    #回答送信(デフォルト)
    def send_ans_default(textbox):
        send_data = textbox.get("1.0", "end-1c")
        print(send_data)

    #トークンとURL確認ウィンドウ
    def token_confirm_func():
        msg = TOKEN + '\n' + SV_URL
        messagebox.showinfo("弊チームのトークンとサーバURLの確認",msg)

#### メインウィンドウの設定 ####
    frm = tk.Tk()

    frm.geometry('600x500')
    frm.title(window_title)
    frm.configure(bg='black')

    #GUIにでっかくタイトルを表示
    big_tit = tk.Label(frm,text=window_title,fg='#77ff77',bg='black',font=("normal",30,"bold","italic"))
    big_tit.place(x=0,y=0)

    #分割データの取得数指定の入力欄
    how_many_data = tk.Label(frm,font=("normal",20),text='分割データ取得数の入力',foreground='#ffffff',background='#000000')
    how_many_data.place(x=0,y=140)
    how_many_data_box = tk.Entry(frm,width=3,justify="center", font=("HGP行書体",20))
    how_many_data_box.place(x=320,y=140)

    #取得したデータのファイル名一覧表示
        #エラー回避のための配列定義
    filepathlst = [] * 5
    got_file_name = tk.Label(frm,font=("normal",15),text='取得した分割データ一覧:',bg="#000000",fg="#ffffff")
    got_file_name.place(x=0,y=200)
    got_file_name_list1 = tk.Entry(frm,width=70)
    got_file_name_list1.place(x=0,y=240)
    got_file_name_list2 = tk.Entry(frm,width=70)
    got_file_name_list2.place(x=0,y=260)
    got_file_name_list3 = tk.Entry(frm,width=70)
    got_file_name_list3.place(x=0,y=280)
    got_file_name_list4 = tk.Entry(frm,width=70)
    got_file_name_list4.place(x=0,y=300)
    got_file_name_list5 = tk.Entry(frm,width=70)
    got_file_name_list5.place(x=0,y=320)

    #ファイルパス一覧の初期化ボタン
    clearlist_btn = tk.Button(frm,font=("normal",15),background='#0077bb',fg='#ffffff',text='パスのリセット',command=clear_path_list)
    clearlist_btn.place(x=400,y=190)

    #チームトークン・URLの確認
    token_confirm = tk.Label(frm,font=("normal",20),foreground='#ffffff',background='#000000',text='URLと弊チームのトークン: ')
    token_confirm.place(x=0,y=70)
        #トークン・URL確認ボタン
    token_confirm_btn = tk.Button(frm,font=("normal",16),background='#00dd44',text='確認',command=token_confirm_func)
    token_confirm_btn.place(x=350,y=70)

    #データ取得ボタンの定義と配置
    get_prob_data_btn = tk.Button(frm,font=("normal",20),background='#ffbb44',text='問題を取得',command=partial(get_prob_data,how_many_data_box))
    get_prob_data_btn.place(x=10,y=430)

    #ローカルファイル参照ボタン
    get_prob_data_local_btn = tk.Button(frm,font=("normal",20),background='#ffbb44',text='ファイルを参照',command=get_prob_data_local)
    get_prob_data_local_btn.place(x=10,y=360)

    #解析開始ボタンの定義と配置
    start_main_program_btn = tk.Button(frm,background='#ff4444',text='解析開始!!',font=("normal",20,"bold"),command=call_main_prog)
    start_main_program_btn.place(x=200,y=430)

    frm.mainloop()

    if os.path.isfile('pathlist') == True:
        os.remove('pathlist')
    else:
        pass