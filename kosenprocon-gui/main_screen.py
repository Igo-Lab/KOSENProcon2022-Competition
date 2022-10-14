import os
import tkinter

os.environ["DISPLAY"] = ":0.0"
# Tkクラス生成
frm = tkinter.Tk()
# 画面サイズ
frm.geometry("600x400")
# 画面タイトル
frm.title("サンプル画面")
# 画面をそのまま表示
frm.mainloop()
