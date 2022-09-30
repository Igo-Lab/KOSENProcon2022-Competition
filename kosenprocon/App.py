import ctypes
from logging import Logger
import os
import time
import wave
import numpy as np
from . import libs
import soxr


class App:
    logger: Logger = None
    compressed_srcs: list[np.ndarray] = []
    raw_src: list[np.ndarray] = []
    compressing_rate: int = libs.constant.COMPRESSING_RATE

    @classmethod
    def app(cls, logger: Logger):
        cls.logger = logger
        logger.info("Starting...")

        cls.load_srcs()

        while True:
            cls.wait_for_it()

    @classmethod
    def load_srcs(cls):
        cls.logger.info("Loading Srcs")
        cls.raw_src.clear()
        cls.compressed_srcs.clear()

        for i in range(1, libs.constant.LOAD_BASE_NUM + 1):
            with wave.open(rf"{libs.constant.BASE_AUDIO_DIR}/{i}.wav") as wr:
                data = np.frombuffer(wr.readframes(-1), dtype=np.int16)
                cls.raw_src.append(data)
                cls.compressed_srcs.append(cls.compress(data))

        cls.logger.info("Loading has been done")

    @classmethod
    def load_problem(cls):
        # とりあえず一旦こうする．本番ではhttpでゲットする仕組みが必要
        pass

    @classmethod
    def compress(cls, src: np.ndarray, rate: int = compressing_rate):
        cls.logger.info(f"Compressing... {len(src)}->{int(len(src)/rate)}")
        rs = soxr.resample(
            src,
            libs.constant.SRC_SAMPLE_RATE,
            int(libs.constant.SRC_SAMPLE_RATE / cls.compressing_rate),
        )
        return rs

    @classmethod
    def set_compaction_rate(cls, rate: int):
        if cls.compressing_rate != rate:
            cls.compressed_srcs.clear()
            for src in cls.raw_src:
                cls.compressed_srcs.append(cls.compress(src))

    @classmethod
    def wait_for_it(cls):
        cls.logger.info("Waiting for match starting.")
        # 今はsleepかけてるだけだけど本番はhttpとってこなきゃいけない
        time.sleep(1)

        cls.logger.warn("===!START!===")

    @classmethod
    def resolve(cls, problem: np.ndarray):
        pass

    @classmethod
    def show_result(cls):
        pass


