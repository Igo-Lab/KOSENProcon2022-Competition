import sys
import urllib.parse
import requests
from pydantic import BaseModel

class AnswerData(BaseModel):
    problem_id: str
    answers: list[str]

def ask_yn(msg) -> bool:
	while True:
		choice = input(msg).lower()
		if choice in ["y", "ye", "yes"]:
			return True
		elif choice in ["n", "no"]:
			return False

REQ_HEADER={
	"procon-token":"214756da1785484c9e578de68f9b58d421e8217816dc863cb8d28dd1274aac41",
	"Content-Type": "application/json",
	"charset": "utf-8",
}

PROXY = {
    "http": "",
    "https": "",
}

r = requests.get(
	urllib.parse.urljoin("http://172.28.1.1:80", "test"),
	headers=REQ_HEADER,
	proxies=PROXY
)

r.raise_for_status()

if r.text == "OK":
	print("Connection OK.")
else:
	print("Not OK.")
	print(r.text)

####
pid = input("Input Mondai ID: ")
ids = input("Input Yomi Data ID(,kugiri)")

id_i: set[int] = set()
for id_s in ids.split(","):
	idx = int(id_s)
	id_i.add(idx)

if ask_yn(f"tugi no yomi-id wo soushin suruyo: answer={id_i}, problem_id={pid}"):
	li = [f"{(x-1)%44+1:02}" for x in id_i]
	ad = AnswerData(problem_id=problem_id, answers=li)
	r = requests.post(
		urllib.parse.urljoin("http://172.28.1.1:80", "problem"),
		headers=REQ_HEADER,
		proxies=PROXY,
		data=ad.json(),
	)

	r.raise_for_status()
	print("DONE!(SEIKO!)")
else:
	print("Cancelled")






