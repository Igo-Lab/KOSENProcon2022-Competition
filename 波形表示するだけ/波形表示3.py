import librosa
import librosa.display

a, sr = librosa.load('E01.wav')
librosa.display.waveshow(a, sr=sr)

import matplotlib.pyplot as plt
plt.show()