import os
import tkinter as tk


class App:

    main_win: tk.Tk

    @classmethod
    def app(cls):
        cls.main_win = tk.Tk()
        cls.main_win.title("春光台にそそりたつ！")
        cls.main_win.configure(bg="#212121")

        cls.main_win.mainloop()
