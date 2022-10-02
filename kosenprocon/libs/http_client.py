from email import header
from pydantic import BaseModel
from typing import *
import constant
import requests
import numpy as np
import wave

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
    r = requests.get(
        constant.API_URL + "/problem", headers=REQ_HEADER, timeout=constant.TIMEOUT
    )
    r.raise_for_status()
    return ProblemData.parse_raw(r.text)


# 重複しているデータは取り寄せたくない
def get_chunks(n: int, already: set[int]) -> np.ndarray[np.ndarray]:
    r = requests.post(
        constant.API_URL + "/problem/chunks",
        params={"n": n},
        headers=REQ_HEADER,
        timeout=constant.TIMEOUT,
    )
    r.raise_for_status()
    chds = ChunkPlaceData.parse_raw(r.text)

    wavs: np.ndarray[np.ndarray] = np.empty(0, dtype=np.ndarray)
    for chd in chds.chunks:
        r = requests.get(
            constant.API_URL + f"/problem/chunks/{chd}",
            headers=REQ_HEADER,
            timeout=constant.TIMEOUT,
        )
        r.raise_for_status()

        with wave.open(r.content) as wr:
            np.append(wavs, np.frombuffer(wr.readframes(-1), dtype=np.int16))

    return wavs


def send_answer():
    pass
