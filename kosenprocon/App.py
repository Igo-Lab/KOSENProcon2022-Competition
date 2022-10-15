import copy
import math
import os
import sys
import time
import wave

import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel
from requests import exceptions

from kosenprocon.libs.constant import FILTER_THRESHOLD
from kosenprocon.libs.http_client import ProblemData

from . import libs


# ファイル名=問題ID.jsonのファイルを読み込み、再開するときに使う
class ResumeData(BaseModel):
    answer: set[int]
    mask: list[bool]


class App:
    # 2D
    compressed_srcs: npt.NDArray[np.int16]
    compressed_srcs_len: npt.NDArray[np.int32]
    raw_srcs: list[list[np.int16]]
    # 1D
    raw_chunks: list[tuple[int, list[np.int16]]] = []
    compressed_chunk: npt.NDArray[np.int16]

    compressing_rate: int = libs.COMPRESSING_RATE

    # 試合データ
    match: libs.MatchData
    problems_data: list[libs.ProblemData] = []

    answer: set[int] = set()
    srcs_mask: npt.NDArray[np.bool_] = np.full(
        libs.LOAD_BASE_NUM, False, dtype=np.bool_
    )  # CUDA処理側に処理しない元データ情報を渡す

    # 復帰処理
    got_num: int = 0

    # gpuから帰ってくるresult
    result_gpu: npt.NDArray[np.uint32] = np.empty(
        (libs.LOAD_BASE_NUM, 2), dtype=np.uint32
    )

    @classmethod
    def app(cls):
        np.set_printoptions(precision=3)

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
                        time.sleep(5)
                        continue

                    # 末尾に問題情報を追加
                    cls.problems_data.append(p)

                    # 各種初期化
                    cls.answer.clear()
                    cls.raw_chunks.clear()
                    cls.got_num = 0
                    resfile_path = f"resume/{cls.problems_data[-1].id}.json"

                    # 本当は時刻チェックでwaitかける処理が必要だが・・・

                    # 復帰処理
                    if cls.ask_yn("前回途中で処理を中断しましたか？ [y/N]: "):
                        cls.got_num = int(input("前回何分割まで取り寄せましたか？: "))
                        for _ in range(cls.got_num):
                            cls.raw_chunks.append(libs.get_chunk(cls.raw_chunks))

                        logger.info("復帰分データ投入完了。")

                        if os.path.exists(resfile_path):
                            logger.info("resumeファイルから復元します。")
                            with open(resfile_path) as f:
                                res = ResumeData.parse_raw(f.read())
                                cls.answer = res.answer
                                cls.srcs_mask = np.array(res.mask, dtype=np.bool_)
                        elif cls.ask_yn("手動で解答を追加しますか？ [y/N]: "):
                            ans: str = input("解答読み札IDを入力して下さい（,区切り）: ")
                            cls.answer = {int(x) for x in ans.split(",")}
                            cls.makemask(cls.answer, cls.srcs_mask)
                            logger.info("解答データ投入完了。")

                    # while文中で確定でファイルを取り寄せるためこのような条件式になっている
                    for _ in range(cls.got_num, cls.problems_data[-1].chunks):
                        cls.raw_chunks.append(libs.get_chunk(cls.raw_chunks))
                        # 1こずつ取り出し、つなげられそうならつなげる
                        tmp_joined_chunk = copy.copy(cls.raw_chunks[-1][1])
                        head = tail = cls.raw_chunks[-1][0]

                        while True:
                            front_element = list(
                                filter(lambda x: x[0] == head - 1, cls.raw_chunks[:-1])
                            )
                            back_element = list(
                                filter(lambda x: x[0] == tail + 1, cls.raw_chunks[:-1])
                            )

                            if len(front_element) > 0:
                                logger.debug(f"前方連結: {head-1}+{head}")
                                fre = front_element[0][1]
                                tmp_joined_chunk = [*fre, *tmp_joined_chunk]
                                head -= 1
                            elif len(back_element) > 0:
                                logger.debug(f"後方連結: {tail}+{tail+1}")
                                bae = back_element[0][1]
                                tmp_joined_chunk = [*tmp_joined_chunk, *bae]
                                tail += 1
                            else:
                                break

                        cls.compressed_chunk = cls.compress(
                            np.array(tmp_joined_chunk, dtype=np.int16),
                            cls.compressing_rate,
                        )

                        # print("chunk ", cls.compressed_chunk[:100])
                        sums = cls.get_sums(
                            cls.compressed_chunk,
                            cls.srcs_mask,
                        )

                        sums = sums[np.argsort(sums[:, 1])]

                        cls.choose_contained(
                            sums, cls.problems_data[-1], cls.srcs_mask, cls.answer
                        )

                        # 結果によってbreakでfor抜ける
                        if len(cls.answer) >= cls.problems_data[-1].data and cls.ask_yn(
                            "結果を送信してもよいですか？ [y/N]: "
                        ):
                            break

                        logger.info("十分なデータが得られなかったため、追加の分割データを取得します。")
                        # 次のCUDA処理のためのマスクを作成
                        cls.makemask(cls.answer, cls.srcs_mask)

                        with open(resfile_path, "w") as f:
                            res = ResumeData(
                                answer=cls.answer, mask=cls.srcs_mask.tolist()
                            )
                            f.write(res.json())

                    # <--break
                    libs.send_answer(cls.problems_data[-1].id, cls.answer)
            except exceptions.RequestException as e:
                logger.warning(f"SERVER ACCESS ERROR. {str(e.response.text)[:-1]}")
                print(e)
                time.sleep(2)

    @classmethod
    def load_srcs(cls, reload=False):
        logger.info("Loading Srcs")

        if not reload:
            cls.raw_srcs = []

            for i in range(1, libs.LOAD_BASE_NUM + 1):
                with wave.open(rf"{libs.BASE_AUDIO_DIR}/{i}.wav") as wr:
                    data = np.frombuffer(wr.readframes(-1), dtype=np.int16)
                    cls.raw_srcs.append(data)

        maxlen = max(len(x) for x in cls.raw_srcs)
        compedarr = []
        lenarr = []

        first = True
        for src in cls.raw_srcs:
            rs = cls.compress(src, cls.compressing_rate)

            if first:
                # print(rs[:100])
                first = False

            lenarr.append(len(rs))  # 元の長さを保存
            rs.resize(
                math.ceil(maxlen / cls.compressing_rate), refcheck=False
            )  # 一番大きい長さに統一
            compedarr.append(rs)

        cls.compressed_srcs = np.array(compedarr, dtype=np.int16)
        cls.compressed_srcs_len = np.array(lenarr, dtype=np.int32)

        libs.memcpy_src2gpu(cls.compressed_srcs, cls.compressed_srcs_len)
        logger.info("Loading has been done")

    @classmethod
    def compress(
        cls, src: np.ndarray, rate: int = compressing_rate
    ) -> npt.NDArray[np.int16]:
        if np.size(src) == 0:
            logger.warning("This src is empty. Check around.")
            raise ValueError

        rs = np.array(src[::rate], dtype=np.int16)
        logger.debug(f"Compressing... {len(src)}->{len(rs)}")

        return rs

    # chunk: 解きたい問題データ
    # srcs: 成形された元データ集
    # src_lengths: 元データの本当の長さ。C側で利用される。
    # mask: 処理しないやつを渡す
    # return: 解析結果（和）のリスト [~][1]には1.wavを元データとして比較したときの最小時の和が入っている。
    #         maskで指定されたものにはUINT_MAXが入る
    @classmethod
    def get_sums(
        cls,
        chunk: npt.NDArray[np.int16],
        mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.uint32]:
        logger.info("Start Processing on CUDA.")

        libs.resolver(chunk, len(chunk), mask, cls.result_gpu)

        return cls.result_gpu

    @classmethod
    def choose_contained(
        cls,
        sums: npt.NDArray[np.uint32],
        pd: libs.ProblemData,
        mask: npt.NDArray[np.bool_],
        answer: set[int],
    ):
        filtered = sums[pd.data : libs.LOAD_BASE_NUM - np.count_nonzero(mask == True)]
        target = sums[: pd.data]
        std = np.std(filtered[:, 1])
        mean = np.mean(filtered[:, 1])

        zscored = (target[:, 1] - mean) / std

        # ztest = (filtered[:, 1] - mean) / std

        # 最後にanswerに追加して終了
        debug_id = [(x - 1) % 44 + 1 for x in target[:, 0]]
        for num, zvalue in zip(target[:, 0], zscored):
            if abs(zvalue) > libs.FILTER_THRESHOLD:
                answer.add(num)

        logger.info(f"候補以外のzscoreの値:")
        zs_other = np.sort(((filtered[:, 1] - mean) / std))
        print(zs_other)
        logger.info(f"候補のzscoreの値: {zscored}")

        temp_ans = [f"{(x-1)%44+1:02}" for x in answer]
        logger.info(f"暫定的な回答: answer={temp_ans}\n")

    # @classmethod
    # def set_compaction_rate(cls, rate: int):
    #     if cls.compressing_rate != rate:
    #         logger.info("compaction rate was changed.")
    #         cls.compressing_rate = rate

    #         # reload
    #         cls.load_srcs(reload=True)
    #         cls.compressed_chunk = cls.compress(cls.raw_chunks, rate)

    @classmethod
    def makemask(cls, answer: set[int], mask: npt.NDArray[np.bool_]):
        for num in answer:
            num = num - 1
            mask[num] = True
            if num < 44:
                mask[num + 44] = True
            else:
                mask[num - 44] = True

    @classmethod
    def show_result(cls):
        pass

    @classmethod
    def ask_yn(cls, msg) -> bool:
        while True:
            choice = input(msg).lower()
            if choice in ["y", "ye", "yes"]:
                return True
            elif choice in ["n", "no"]:
                return False
