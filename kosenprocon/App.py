import ctypes
import sys
from loguru import logger
import os
import time
import wave
import numpy as np
import numpy.typing as npt
import copy
from . import libs
import soxr
from requests import Timeout, exceptions


class App:
    # 2D
    compressed_srcs: npt.NDArray[np.int16]
    compressed_srcs_len: npt.NDArray[np.int32]
    raw_srcs: list[list[np.int16]]
    # 1D
    raw_chunks: list[tuple[int, list[int]]] = []
    compressed_chunk: npt.NDArray[np.int16]

    compressing_rate: int = libs.COMPRESSING_RATE

    match: libs.MatchData
    problems_data: list[libs.ProblemData] = []

    @classmethod
    def app(cls):
        logger.remove()
        logger.add(
            sys.stdout,
            level=libs.LOG_LEVEL,
        )
        logger.info("Starting...")

        cls.load_srcs()

        # 試合開始待機ループ
        while True:
            try:
                cls.match = libs.get_match()

                # 問題取得ループ
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

                    # 本当は時刻チェックでwaitかける処理が必要だが・・・

                    # chunkファイルを取り寄せる
                    # while文中で確定でファイルを取り寄せるためこのような条件式になっている
                    for _ in range(cls.problems_data[-1].chunks):
                        cls.raw_chunks.append(libs.get_chunk(cls.raw_chunks))
                        # 1こずつ取り出し、つなげられそうならつなげる
                        tmp_joined_chunk = copy.copy(cls.raw_chunks[-1][1])
                        for pre_chunk in cls.raw_chunks[:-1]:
                            # 後ろに連番になってないか。raw_chunks={(1, []), (3, []), (2, [])})のとき、[1]にヒットする。
                            if cls.raw_chunks[-1][0] + 1 == pre_chunk[0]:
                                tmp_joined_chunk = [*tmp_joined_chunk, *pre_chunk[1]]

                            if cls.raw_chunks[-1][0] - 1 == pre_chunk[0]:
                                tmp_joined_chunk = [*pre_chunk[1], *tmp_joined_chunk]

                        cls.compressed_chunk = cls.compress(
                            np.array(tmp_joined_chunk, dtype=np.int16),
                            libs.COMPRESSING_RATE,
                        )

                        cls.solve(
                            cls.compressed_chunk,
                            cls.compressed_srcs,
                            cls.compressed_srcs_len,
                        )

                    # 結果によってbreakでfor抜ける

                    # F5アタックにならないようにリミッタ
                    time.sleep(2)
                time.sleep(2)
            except exceptions.RequestException as e:
                logger.warning(
                    "Can't retrieve a problem. May be the server is over loaded or the match isn't started."
                )
                print(e)
                time.sleep(2)

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
        lenarr = []

        for src in cls.raw_srcs:
            rs = cls.compress(src, cls.compressing_rate)
            rs_copy = rs.copy()  # soxlibからコピー
            rs_copy.resize(
                int(maxlen / cls.compressing_rate), refcheck=False
            )  # 一番大きい長さに統一
            compedarr.append(rs_copy)
            lenarr.append(len(rs))  # 元の長さを保存

        cls.compressed_srcs = np.array(compedarr, dtype=np.int16)
        cls.compressed_srcs_len = np.array(lenarr, dtype=np.int32)
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

    # chunk: 解きたい問題データ
    # srcs: 成形された元データ集
    # src_lengths: 元データの本当の長さ。C側で利用される。
    # return: 解析結果（和）のリスト
    @classmethod
    def solve(
        cls,
        chunk: npt.NDArray[np.int16],
        srcs: npt.NDArray[np.int16],
        src_lengths: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int32]:
        # TODO: cuda呼び出し
        logger.info("Start Processing on CUDA.")
        logger.debug(f"problem chunk size: {chunk.shape}, srcs arr size()")
        pass

    # @classmethod
    # def set_compaction_rate(cls, rate: int):
    #     if cls.compressing_rate != rate:
    #         cls.compressing_rate = rate
    #         cls.compressed_srcs.clear()

    #         for src in cls.raw_srcs:
    #             cls.compressed_srcs.append(cls.compress(src), rate)

    #         cls.compressed_problem = cls.compress(cls.raw_chunks, rate)

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
