from email import header
from hashlib import new
from pydantic import BaseModel
from typing import *
from . import constant
import requests
import numpy as np
import numpy.typing as npt
import wave
from loguru import logger

REQ_HEADER = {"procon-token": constant.API_TOKEN}


class MatchData(BaseModel):
    problems: int
    bonus_factor: List[float]
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
    r = requests.get(
        constant.API_URL + "/match", headers=REQ_HEADER, timeout=constant.TIMEOUT
    )
    r.raise_for_status()
    return MatchData.parse_raw(r.text)


def get_problem() -> ProblemData:
    logger.info("Get a problem.")
    r = requests.get(
        constant.API_URL + "/problem", headers=REQ_HEADER, timeout=constant.TIMEOUT
    )
    r.raise_for_status()
    return ProblemData.parse_raw(r.text)


# 重複しているデータは取り寄せたくない
# すでに持っている問題データとつながりそうならつなげて返却する
# already: 事前に持っているchunksの番号とデータ内容のlist。
# return: (データ番号,取得したデータ)のタプル


def get_chunk(already: list[(int, list[np.int16])]) -> tuple[int, list[int]]:
    r = requests.post(
        constant.API_URL + "/problem/chunks",
        params={"n": len(already) + 1},
        headers=REQ_HEADER,
        timeout=constant.TIMEOUT,
    )
    r.raise_for_status()
    chds = ChunkPlaceData.parse_raw(r.text)

    new_no = int(chds.chunks[-1].spilit("_")[0][-1])

    r = requests.get(
        constant.API_URL + f"/problem/chunks/{chds.chunks[-1]}",
        headers=REQ_HEADER,
        timeout=constant.TIMEOUT,
    )
    r.raise_for_status()

    wav: npt.NDArray[np.int16]
    with wave.open(r.content) as wr:
        wav = np.frombuffer(wr.readframes(-1), dtype=np.int16).tolist()

    return (new_no, wav)


def send_answer():
    pass
