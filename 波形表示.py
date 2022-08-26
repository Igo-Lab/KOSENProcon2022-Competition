import wave
import numpy as np
import matplotlib.pyplot as plt

wf = wave.open("E01.wav" , "r" )
buf = wf.readframes(wf.getnframes())

# バイナリデータを整数型（16bit）に変換
data = np.frombuffer(buf, dtype="int16")

# グラフ化
plt.plot(data)
plt.grid()
plt.show()
