import ctypes
from loguru import logger
import os
import time
import wave
import numpy as np
import libs
import soxr
from requests import Timeout, exceptions


class App:
    compressed_srcs: list[np.ndarray] = []
    raw_src: list[np.ndarray] = []
    raw_problem: np.ndarray = np.array([], dtype=np.int16)
    compressed_problem: np.ndarray = np.array([], dtype=np.int16)
    compressing_rate: int = libs.COMPRESSING_RATE

    match: libs.MatchData
    problems: list[libs.ProblemData] = []

    @classmethod
    def app(cls):
        logger.info("Starting...")

        cls.load_srcs()

        while True:
            try:
                cls.match = libs.get_match()
                while True:
                    p = libs.get_problem()
                    # 事前に配布された問題と被っていないか確認
                    same_id_num = len(
                        [+1 for _ in filter(lambda prob: prob.id == p.id, cls.problems)]
                    )

                    if same_id_num > 0:
                        continue

                    # F5アタックにならないようにストッパ
                    time.sleep(0.5)
            except exceptions.RequestException:
                logger.warning(
                    "Can't retrieve a problem. May be the server is over loaded or not the match isn't started."
                )

    @classmethod
    def load_srcs(cls):
        logger.info("Loading Srcs")
        cls.raw_src.clear()
        cls.compressed_srcs.clear()

        for i in range(1, libs.LOAD_BASE_NUM + 1):
            with wave.open(rf"{libs.BASE_AUDIO_DIR}/{i}.wav") as wr:
                data = np.frombuffer(wr.readframes(-1), dtype=np.int16)
                cls.raw_src.append(data)
                cls.compressed_srcs.append(cls.compress(data))

        logger.info("Loading has been done")

    @classmethod
    def load_problem(cls):
        logger.info("Loading a problem")
        # とりあえず一旦こうする．本番ではhttpでゲットする仕組みが必要
        with wave.open(libs.EXAMPLE_PROBLEM) as wr:
            cls.raw_problem = np.frombuffer(wr.readframes(-1), dtype=np.int16)
            cls.compressed_problem = cls.compress(cls.raw_problem)

    @classmethod
    def compress(cls, src: np.ndarray, rate: int = compressing_rate):
        if np.size(src) == 0:
            logger.warning("This src is empty. Check around.")
            raise FileNotFoundError

        logger.info(f"Compressing... {len(src)}->{int(len(src)/rate)}")
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

            for src in cls.raw_src:
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
