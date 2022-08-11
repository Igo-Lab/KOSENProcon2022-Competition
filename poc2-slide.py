import time
import wave
import numpy as np
import matplotlib.pyplot as plt

problem: str = r"samples\original\problem_2.wav"
src_path: str = r"samples\JKspeech"
frame_size = 0
raw_data_offset = 0
similarity = 0
is_first = True


def main():
    global problem, src_path, frame_size, raw_data_offset, similarity, is_first
    start = time.perf_counter()
    with wave.open(problem, "r") as wr:
        ch = wr.getnchannels()
        width = wr.getsampwidth()
        fr = wr.getframerate()
        frame_size = wr.getnframes()

        print(f"ch: {ch}, width: {width}, fr: {fr}, fn: {frame_size}")

        data = wr.readframes(wr.getnframes())
        problem_data = np.frombuffer(data, dtype=np.int16)
        print(f"sum={np.sum(np.abs(problem_data))}")

    timeline = np.arange(0, frame_size)
    for i in range(1, 3 + 1):
        with wave.open(rf"{src_path}\E{i:02}.wav") as wr:
            is_first = True
            raw_data_offset = 0
            data = wr.readframes(wr.getnframes())
            data = np.frombuffer(data, dtype=np.int16)
            print(f"open: E{i:02}.wav")

            # frameもスキップ数の整数倍に成形しなければいけない
            for j in range(0, wr.getnframes(), 5):
                clipped = data[j : min(wr.getnframes(), j + frame_size)]

                translated = np.resize(clipped, (frame_size,))
                translated[clipped.shape[0] :] = 0

                remaining = problem_data - translated

                # timeline = np.arange(0, 72000)
                # plt.plot(timeline, problem_data)
                # plt.show()

                if is_first:
                    similarity = np.sum(np.abs(remaining))
                    is_first = False
                else:
                    s_candidate = np.sum(np.abs(remaining))
                    if similarity > s_candidate:
                        similarity = s_candidate
                        raw_data_offset = j

            print(f"similarity: {similarity} offset:{raw_data_offset}")

    elapssed_time = time.perf_counter() - start
    print(f"Elapssed Time: {elapssed_time}[s]")
    # print(f"similarity: {similarity} offset:{raw_data_offset}")

    # timeline = np.arange(0, fn)
    # plt.plot(timeline, problem_data)
    # plt.show()


if __name__ == "__main__":
    main()
