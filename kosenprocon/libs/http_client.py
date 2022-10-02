from pydantic import BaseModel
from typing import *
import constant
import requests
import pydantic

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


def get_chunks(n: int):
    pass


def send_answer():
    pass
