#比較元と比較対象で、波形データが同じかどうか確かめるプログラム
#test1.pyをもとにしてるにゃ
#ちなみに、まだうごかない。21行目でエラー。

import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf

def wavload(path):
    data, samplerate = sf.read(path)
    return data, samplerate

def ckwav(d1,d2):
    sourcewv , sam1=sf.read(d1)
    targetwv , sam2=sf.read(d2)

    count=0

    for i in range(sourcewv.size):

        if all(sourcewv[i]) == all(targetwv[i]):
            count+=1

    if count >= 70:
        print("同じ可能性が高い\n")
    else:
        print("ワンちゃん違う音声\n")

#spath = input("比較元のパス\n")
#tpath = input("比較対象のパス\n")

#いちいちパス入れるのだるいんで。
spath = 'samples\JKspeech\E01.wav'
tpath = spath
ckwav(spath,tpath)
