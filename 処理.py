import tkinter as tk
import requests

def pushed():
    res=requests.get(url,param)



root = tk.Tk()

root.geometry("400x300")

root.title("かるたGUI")




button = tk.Button(root, text="wav取得", command=pushed)
button.place(x=90, y=120)

root.mainloop()