import os
import sys
import urllib.parse

import requests
from kosenprocon.libs import constant, http_client


def ask_yn(msg) -> bool:
    while True:
        choice = input(msg).lower()
        if choice in ["y", "ye", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False


if __name__ == "__main__":
    REQ_HEADER = {
        "procon-token": constant.API_TOKEN,
        "Content-Type": "application/json",
        "charset": "utf-8",
    }

    arg1 = sys.argv[1]

    if "-t" in arg1:
        r = requests.get(
            urllib.parse.urljoin(constant.API_URL, "test"),
            headers=REQ_HEADER,
            PROXY=constant.PROXY,
            timeout=3.0,
        )

        r.raise_for_status()

        if r.body == "OK":
            print("Connection OK.")
        else:
            print("Not OK.")
            print(r.body)

    if "-s" in arg1:
        pid = input("問題IDを入力してください: ")
        ids = input("送信する読みデータIDを入力してください: ")

        id_i: set[int] = set()
        for id_s in ids.split(","):
            id = int(id_s)
            id_i.add(id)

        if ask_yn(f"次のIDsを解答として送信します。よろしいですか？ [y/N]:\nanswer={id_i}, problem_id={pid}"):
            http_client.send_answer(pid, id_i)
        else:
            print("中断しました．")
