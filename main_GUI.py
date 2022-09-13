from fileinput import close
from genericpath import isfile
from re import I
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tokenize import String, Token
import numpy as np
from functools import partial
import os

import call_test #仮でimportしている
import post_client

if __name__ == '__main__':

    try:
        with open('config.txt','r',encoding='utf-8') as CONFIG_FILE:
            TOKEN = CONFIG_FILE.readline()
            SV_URL = CONFIG_FILE.readline()
            CONFIG_FILE.close()

    except FileNotFoundError:
        messagebox.showerror('エラー','config.txtが存在しません')

    else:

        window_title = '音声解析の鬼(仮) (GUIテスト)'

        #回答送信の確認画面
        def send_ans(data,list):
            wid = tk.Tk()

            wid.geometry('600x450')
            wid.title('回答送信確認画面')
            wid.configure(bg="black")

            #判別結果の画面出力と、修正画面
            #自動判別結果側
            box_name1 = tk.Label(wid,font=("normal",20),text='判別結果',bg="black",fg="#ffffff")
            box_name1.place(x=0,y=0)
            box_ans = tk.Entry(wid,width=50)
            box_ans.place(x=50,y=50)
            box_ans.insert(tk.END,str(data))
            #修正入力側
            box_name2 = tk.Label(wid,font=("normal",20),text='修正を入力',bg="black",fg="#ffffff")
            box_name2.place(x=0,y=150)
            box_fix = tk.Entry(wid,width=50)
            box_fix.place(x=50,y=200)

            #差分合計値リストの表示
            list_name1 = tk.Label(wid,font=("normal",15),text='差分合計値リスト',bg='black',fg='#ffffff')
            list_name1.place(x=0,y=80)
            sum_list = tk.Label(wid,text=list,bg='black',fg='#ffffff')
            sum_list.place(x=0,y=115)

            #送信ボタン
            send_btn = tk.Button(wid,font=("normal",20,"bold"),text='修正せずに回答を送信',bg="#ff0044",command=partial(post_ans,data))
            send_btn.place(x=140,y=280)


            #修正した回答の送信ボタン
            fixed_send_btn = tk.Button(wid,font=("normal",20,"bold"),text='修正した内容を送信',bg="#00dd44",command=partial(post_fix_ans,box_fix))
            fixed_send_btn.place(x=150,y=350)
            
            #画面下の注意書き
            attention_label = tk.Label(wid,font=("normal",10,"bold"),text='確認ダイアログは出ません!!',bg='red',fg='#ffffff')
            attention_label.place(x=190,y=410)

        #メイン画面に取得したファイル名を表示
        def print_filenames(tko,names):
            tko.insert(tk.END,names)

        #分割データを取得
        def get_prob_data(hmd):
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
                #pathlistにメモした内容を再度読み込み＆解析ロジックにパスを渡す
            try:
                with open('pathlist','r',encoding='utf-8') as r:
                    filepathlst = r.readlines()
                    print(filepathlst)
                    r.close()

            except FileNotFoundError:
                messagebox.showerror('ファイルパス参照エラー','pathlistからパスを読み取れませんでした')
            prb_data , sum_list = call_test.test(filepathlst)
            print('完了')

            send_ans(prb_data,sum_list)

        #判別結果を送信
        def post_ans(answer):
            print('自動判別の回答を送信した')
            print(answer)

        #修正した回答のデータを送信
        def post_fix_ans(func):
            
            print('修正済みの回答を送信した')
            print(func.get())

        #トークンとURL確認ウィンドウ
        def token_confirm_func():
            msg = TOKEN + '\n' + SV_URL
            messagebox.showinfo("弊チームのトークンとサーバURLの確認",msg)

        #メインウィンドウの設定
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
        how_many_data_box = tk.Entry(frm,width=20)
        how_many_data_box.place(x=320,y=150)

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
        token_confirm.place(x=0,y=60)
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
        start_main_program_btn = tk.Button(frm,background='#ff0044',text='解析開始!!',font=("normal",20,"bold"),command=call_main_prog)
        start_main_program_btn.place(x=200,y=430)

        frm.mainloop()

        if os.path.isfile('pathlist') == True:
            os.remove('pathlist')
        else:
            pass