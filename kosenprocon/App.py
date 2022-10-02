import ctypes
import sys
from loguru import logger
import os
import time
import wave
import numpy as np
import numpy.typing as npt
import libs
import soxr
from requests import Timeout, exceptions


class App:
    # 2D
    compressed_srcs: npt.NDArray[np.int16]
    raw_srcs: list[list[np.int16]]
    # 1D
    raw_problem: list[list[np.int16]]
    compressed_problem: npt.NDArray[np.int16]

    compressing_rate: int = libs.COMPRESSING_RATE

    match: libs.MatchData
    problems_data: list[libs.ProblemData] = []
    already_have_problem: set[int] = set()

    @classmethod
    def app(cls):
        logger.add(
            sys.stdout,
            level=libs.LOG_LEVEL,
        )
        logger.info("Starting...")

        cls.load_srcs()

        while True:
            try:
                cls.match = libs.get_match()
                while True:
                    p = libs.get_problem()
                    # 事前に配布された問題と被っていないか確認
                    same_id_num = len(
                        [
                            +1
                            for _ in filter(
                                lambda prob: prob.id == p.id, cls.problems_data
                            )
                        ]
                    )

                    if same_id_num > 0:
                        continue

                    # 末尾に問題情報を追加
                    cls.problems_data.append(p)

                    # 問題ファイルを取り寄せる
                    while cls.problems_data[-1].chunks > len(cls.already_have_problem):
                        l = libs.get_chunks(1, cls.already_have_problem)

                    # F5アタックにならないようにリミッタ
                    time.sleep(0.5)
            except exceptions.RequestException:
                logger.warning(
                    "Can't retrieve a problem. May be the server is over loaded or not the match isn't started."
                )

    @classmethod
    def load_srcs(cls):
        logger.info("Loading Srcs")
        cls.raw_srcs = []

        for i in range(1, libs.LOAD_BASE_NUM + 1):
            with wave.open(rf"{libs.BASE_AUDIO_DIR}/{i}.wav") as wr:
                data = np.frombuffer(wr.readframes(-1), dtype=np.int16)
                cls.raw_srcs.append(data)

        maxlen = max(len(x) for x in cls.raw_srcs)
        compedarr = []

        for src in cls.raw_srcs:
            rs = cls.compress(src, cls.compressing_rate)
            rs.resize(int(maxlen / cls.compressing_rate))  # 一番大きい長さに統一
            compedarr.append(rs)

        cls.compressed_srcs = np.array(compedarr, dtype=np.int16)
        logger.info("Loading has been done")

    @classmethod
    def compress(cls, src: np.ndarray, rate: int = compressing_rate):
        if np.size(src) == 0:
            logger.warning("This src is empty. Check around.")
            raise ValueError

        logger.debug(f"Compressing... {len(src)}->{int(len(src)/rate)}")
        rs = soxr.resample(
            src,
            libs.SRC_SAMPLE_RATE,
            int(libs.SRC_SAMPLE_RATE / cls.compressing_rate),
        )
        return rs

    @classmethod
    def set_compaction_rate(cls, rate: int):
        if cls.compressing_rate != rate:
            cls.compressing_rate = rate
            cls.compressed_srcs.clear()

            for src in cls.raw_srcs:
                cls.compressed_srcs.append(cls.compress(src), rate)

            cls.compressed_problem = cls.compress(cls.raw_problem, rate)

    @classmethod
    def wait_for_match(cls):
        logger.info("Waiting for match starting.")
        # 今はsleepかけてるだけだけど本番はhttpとってこなきゃいけない
        time.sleep(1)

        logger.warning("===!START!===")

    @classmethod
    def resolve(cls, problem: np.ndarray):
        pass

    @classmethod
    def show_result(cls):
        pass
