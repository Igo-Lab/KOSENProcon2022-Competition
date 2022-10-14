import os
import tkinter as tk

BG_COLOR = "#212121"
FONT_COLOR = "#e5e5e5"

S_DIMX = 1920
S_DIMY = 1080
W_WIDTH = 1024
W_HEIGHT = 768

S_CENTERX = S_DIMX // 2 - W_WIDTH // 2
S_CENTERY = S_DIMY // 2 - W_HEIGHT // 2


class App:

    main_win: tk.Tk

    @classmethod
    def app(cls):
        cls.main_win = tk.Tk()

        cls.main_win.geometry(f"{W_WIDTH}x{W_HEIGHT}+{S_CENTERX}+{S_CENTERY}")
        cls.main_win.resizable(False, False)
        cls.main_win.title("春光台にそそりたつ！")
        cls.main_win.configure(bg=BG_COLOR)

        # タイトル
        title = tk.Label(
            cls.main_win,
            text="春光台にそそりたつ",
            fg=FONT_COLOR,
            bg=BG_COLOR,
            font=("normal", 30, "bold"),
        )
        title.place(x=0, y=0)

        txt = tk.Entry(width=20)
        txt.place(x=0, y=200)

        cls.main_win.mainloop()
