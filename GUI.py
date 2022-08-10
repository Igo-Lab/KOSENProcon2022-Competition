import tkinter
from tkinter import messagebox
import subprocess

def fun1():
    #messagebox.showwarning('おい！！！','押すなって言ったよね？？？')
    subprocess.call('called_by_GUI.py_.exe')

frm = tkinter.Tk()

frm.geometry('300x200')
frm.title('挑戦')

btn = tkinter.Button(frm,text='絶対押すなよ！！',command=fun1)
btn.place(x=130,y=80)

frm.mainloop()