import librosa

file_name = 'E01.wav'
wav, sr = librosa.load(file_name, sr=44100)

import librosa.display
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
librosa.display.waveshow(wav, sr)
plt.show()