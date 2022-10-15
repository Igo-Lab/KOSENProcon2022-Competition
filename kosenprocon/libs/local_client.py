import uuid
import wave
from typing import *

import numpy as np
from loguru import logger
from pydantic import BaseModel

from . import constant

REQ_HEADER = {
    "procon-token": constant.API_TOKEN,
    "Content-Type": "application/json",
    "charset": "utf-8",
}
PROXY = constant.PROXY


class MatchData(BaseModel):
    problems: int
    bonus_factor: list[float]
    penalty: int


class ProblemData(BaseModel):
    id: str
    chunks: int
    starts_at: int
    time_limit: int
    data: int


class ChunkPlaceData(BaseModel):
    chunks: list[str]


class AnswerData(BaseModel):
    problem_id: str
    answers: list[str]


class answer_verified_data(BaseModel):
    problem_id: str
    answers: list[str]
    accepted_at: int


def get_match() -> MatchData:
    logger.info("Trying to get a match.")
    problem_num: int = 0

    return MatchData(problems=problem_num, bonus_factor=[], penalty=0)


def get_problem() -> ProblemData:
    logger.info("Trying to get a problem.")

    print("1. 問題のデータを入力してください。: ")
    problem_id: str = input("問題IDを入力してください: ")
    chunks_num: int = int(input("分割データの個数を入力してください。: "))
    overlap: int = int(input("重ね合わせ数を入力してください。: "))

    return ProblemData(
        id=problem_id, chunks=chunks_num, starts_at=0, time_limit=0, data=overlap
    )


# already: 事前に持っているchunksの番号とデータ内容のlist。
# return: (データ番号,取得したデータ)のタプル


def get_chunk(already: list[tuple[int, list[np.int16]]]) -> tuple[int, list[np.int16]]:
    path = input("分割データのパスを入力してください: ")
    order: int = int(input("このチャンクの並び順を入力してください: "))

    wav: list[np.int16]
    with wave.open(path) as wr:
        wav = np.frombuffer(wr.readframes(-1), dtype=np.int16).tolist()

    return (order, wav)


def send_answer(problem_id: str, answer: set[int]):
    logger.info("Trying to sending the answer.")
    li = [f"{(x-1)%44+1:02}" for x in answer]

    ad = AnswerData(problem_id=problem_id, answers=li)

    logger.info(f"2. 送信完了。 problem-id={problem_id} answer={li}")
    input("Enterキーを押すと続行します・・・")
