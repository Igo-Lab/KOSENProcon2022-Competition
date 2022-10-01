from pydantic import BaseModel
from typing import *


class match_data(BaseModel):
    problems: int
    bonus_factor: List[float]
    penalty: int


class problem_data(BaseModel):
    id: str
    chunks: int
    starts_at: int
    time_limit: int
    data: int


class chunk_place_data(BaseModel):
    chunks: list[str]


class answer_data(BaseModel):
    problem_id: str
    answers: list[str]


class answer_verified_data(BaseModel):
    problem_id: str
    answers: list[str]
    accepted_at: int


def get_match():
    pass


def get_problem():
    pass


def get_chunks(n: int):
    pass


def send_answer():
    pass
