import time
import wave
import numpy as np
import matplotlib.pyplot as plt

problem = r"samples\sample_Q_202205\sample_Q_E01\problem.wav"
src_path = r"C:\Users\takei\Documents\KOSENProcon\dev\samples\JKspeech"
Elements = ["E01.wav", "E02.wav", "E03.wav"]
remainings = []


def main():
    start = time.perf_counter()
    with wave.open(problem, "r") as wr:
        ch = wr.getnchannels()
        width = wr.getsampwidth()
        fr = wr.getframerate()
        fn = wr.getnframes()

        print(f"ch: {ch}, width: {width}, fr: {fr}, fn: {fn}")

        data = wr.readframes(wr.getnframes())
        problem_data = np.frombuffer(data, dtype=np.int16)

    for i in range(1, 44 + 1):
        with wave.open(rf"{src_path}\E{i:02}.wav") as wr:
            data = wr.readframes(wr.getnframes())
            data = np.frombuffer(data, dtype=np.int16)
            data = np.resize(data, (problem_data.shape[0]))

            remaining = problem_data - data
            remainings.append((i, np.sum(np.abs(remaining))))

    remainings.sort(key=lambda x: x[1])
    elapssed_time = time.perf_counter() - start
    print(f"Elapssed Time: {elapssed_time}[s]")
    for e in remainings:
        print(f"E{e[0]:02}.wav : similarity={e[1]}")

    # timeline = np.arange(0, fn)
    # plt.plot(timeline, problem_data)
    # plt.show()


if __name__ == "__main__":
    main()
