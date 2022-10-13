import io
import urllib.parse
import wave
from typing import *

import numpy as np
import requests
from loguru import logger
from pydantic import BaseModel

from . import constant

REQ_HEADER = {"procon-token": constant.API_TOKEN}
PROXY = {"http": "", "https": ""}


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
    r = requests.get(
        urllib.parse.urljoin(constant.API_URL, "/match"),
        headers=REQ_HEADER,
        timeout=constant.TIMEOUT,
        proxies=PROXY,
    )
    r.raise_for_status()
    return MatchData.parse_raw(r.text)


def get_problem() -> ProblemData:
    logger.info("Trying to get a problem.")
    r = requests.get(
        urllib.parse.urljoin(constant.API_URL, "/problem"),
        headers=REQ_HEADER,
        timeout=constant.TIMEOUT,
        proxies=PROXY,
    )
    r.raise_for_status()
    return ProblemData.parse_raw(r.text)


# already: 事前に持っているchunksの番号とデータ内容のlist。
# return: (データ番号,取得したデータ)のタプル


def get_chunk(already: list[tuple[int, list[np.int16]]]) -> tuple[int, list[np.int16]]:
    r = requests.post(
        urllib.parse.urljoin(constant.API_URL, "/problem/chunks"),
        params={"n": len(already) + 1},
        headers=REQ_HEADER,
        timeout=constant.TIMEOUT,
        proxies=PROXY,
    )
    r.raise_for_status()
    chds = ChunkPlaceData.parse_raw(r.text)

    new_no = int(chds.chunks[-1].split("_")[0][-1])
    logger.info(f"Trying to get chunk={new_no}")
    r = requests.get(
        urllib.parse.urljoin(constant.API_URL, f"/problem/chunks/{chds.chunks[-1]}"),
        headers=REQ_HEADER,
        timeout=constant.TIMEOUT,
        proxies=PROXY,
    )
    r.raise_for_status()

    wav: list[np.int16]
    with wave.open(io.BytesIO(r.content)) as wr:
        wav = np.frombuffer(wr.readframes(-1), dtype=np.int16).tolist()

    return (new_no, wav)


def send_answer(problem_id: str, answer: set[int]):
    logger.info("Trying to sending the answer.")
    li = [f"{x:02}" for x in answer]

    ad = AnswerData(problem_id=problem_id, answers=li)

    r = requests.post(
        urllib.parse.urljoin(constant.API_URL, "/problem"),
        headers=REQ_HEADER,
        timeout=constant.TIMEOUT,
        proxies=PROXY,
        json=ad.json(),
    )
    r.raise_for_status()

    logger.info(f"Sending complete. problem-id={problem_id} answer={answer}")
