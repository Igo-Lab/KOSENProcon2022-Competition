import soundfile as sf
def wav_read(path):
    wave, fs = sf.read(path) #音データと周波数を読み込む
    return wave, fs

fpath =input("gulg")

waved , fsd =wav_read(fpath)

print(waved)
print(fsd)