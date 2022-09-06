import io
import urllib.parse
import wave
from typing import *

import numpy as np
import requests
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

    print("Input Match Data:")
    problem_num: int = int(input("Problems num: "))

    return MatchData(problems=problem_num, bonus_factor=[], penalty=0)


def get_problem() -> ProblemData:
    logger.info("Trying to get a problem.")

    print("Input Problem Data:")
    problem_id: str = input("Problem ID: ")
    chunks_num: int = int(input("Chunks Num: "))
    overlap: int = int(input("Overlap(data) Num:"))

    return ProblemData(
        id=problem_id, chunks=chunks_num, starts_at=0, time_limit=0, data=overlap
    )


# already: 事前に持っているchunksの番号とデータ内容のlist。
# return: (データ番号,取得したデータ)のタプル


def get_chunk(already: list[tuple[int, list[np.int16]]]) -> tuple[int, list[np.int16]]:
    path = input("Please input a chunk path.")
    order: int = int(input("Please input the order of this chunk."))

    wav: list[np.int16]
    with wave.open(path) as wr:
        wav = np.frombuffer(wr.readframes(-1), dtype=np.int16).tolist()

    return (order, wav)


def send_answer(problem_id: str, answer: set[int]):
    logger.info("Trying to sending the answer.")
    li = [f"{x:02}" for x in answer]

    ad = AnswerData(problem_id=problem_id, answers=li)

    logger.info(f"Sending complete. problem-id={problem_id} answer={answer}")
    input("Please press enter key...")
