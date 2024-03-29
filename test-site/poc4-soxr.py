import time
import wave
import numpy as np
import soxr

problem: str = r"samples\sample_Q_202205\sample_Q_M01\problem.wav"
src_path: str = r"samples\JKspeech"
frame_size = 0
raw_data_offset = 0
similarity = 0
is_first = True
translated_sum = 0
similarity_list = []
SKIP_N = 16


def main():
    global problem, src_path, frame_size, raw_data_offset, similarity, is_first, translated_sum, similarity_list
    start = time.perf_counter()
    with wave.open(problem, "r") as wr:
        ch = wr.getnchannels()
        width = wr.getsampwidth()
        fr = wr.getframerate()
        frame_size = wr.getnframes()

        print(f"ch: {ch}, width: {width}, fr: {fr}, fn: {frame_size}")

        data = wr.readframes(-1)
        problem_data = np.frombuffer(data, dtype=np.int16)

        mean_problem_data = soxr.resample(
            problem_data, in_rate=48000, out_rate=48000 / SKIP_N, quality="MQ"
        )
        print(f"sum={np.sum(np.abs(problem_data))}")

    timeline = np.arange(0, frame_size)
    for i in range(1, 88 + 1):
        with wave.open(rf"{src_path}\{i}.wav") as wr:
            is_first = True
            raw_data_offset = 0
            data = wr.readframes(-1)
            data = np.frombuffer(data, dtype=np.int16)
            print(f"open: {i}.wav")

            mean_data = soxr.resample(
                data, in_rate=48000, out_rate=48000 / SKIP_N, quality="MQ"
            )

            for j in range(1, mean_problem_data.__len__() + mean_data.__len__(), 1):
                clip_starti = max(0, j - mean_data.__len__())
                clip_endi = min(j, mean_problem_data.__len__())
                data_starti = max(mean_data.__len__() - j, 0)
                data_endi = min(
                    mean_data.__len__(),
                    mean_data.__len__() + mean_problem_data.__len__() - j,
                )
                # print(j, clip_starti, clip_endi, data_starti, data_endi)
                subbed = (
                    mean_problem_data[clip_starti:clip_endi]
                    - mean_data[data_starti:data_endi]
                )
                remaining = (
                    np.sum(np.abs(subbed))
                    + np.sum(np.abs(mean_problem_data[0:clip_starti]))
                    + np.sum(
                        np.abs(mean_problem_data[clip_endi : len(mean_problem_data)])
                    )
                )

                if is_first:
                    similarity = remaining
                    is_first = False
                else:
                    if similarity > remaining:
                        similarity = remaining
                        raw_data_offset = j

            print(f"similarity: {similarity} offset:{raw_data_offset}")
            similarity_list.append((i, similarity))

    similarity_list.sort(key=lambda x: x[1])
    for e in similarity_list:
        print(f"J{e[0]:02}.wav similarity={e[1]}")
    elapssed_time = time.perf_counter() - start
    print(f"Elapsed Time: {elapssed_time}[s]")

    # timeline = np.arange(0, fn)
    # plt.plot(timeline, problem_data)
    # plt.show()


if __name__ == "__main__":
    main()
