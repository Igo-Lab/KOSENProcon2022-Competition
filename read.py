import struct
import wave
import numpy as np
import matplotlib.pyplot as plt


def main():
    wavf = "samples\sample_Q_202205\sample_Q_E01\problem.wav"
    with wave.open(wavf, "r") as wr:
        ch = wr.getnchannels()
        width = wr.getsampwidth()
        fr = wr.getframerate()
        fn = wr.getnframes()

        print(f"ch: {ch}, width: {width}, fr: {fr}, fn: {fn}")

        data = wr.readframes(wr.getnframes())
        wav_data = np.frombuffer(data, dtype=np.int16)

    timeline = np.arange(0, fn)

    plt.plot(timeline, wav_data)
    plt.show()


if __name__ == "__main__":
    main()
