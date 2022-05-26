#比較元と比較対象で、波形データが同じかどうか確かめるプログラム

import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf

def wavload(path):
    data, samplerate = sf.read(path)
    return data, samplerate

def ckwav(d1,d2):
    sourcewv , sam1=sf.read(d1)
    targetwv , sam2=sf.read(d2)

    if all(sourcewv) == all(targetwv):
        print("こいつぁ、いいぜぇ...\n")
    else:
        print("よくないね。\n")


spath = input("比較元のパス\n")
tpath = input("比較対象のパス\n")

ckwav(spath,tpath)
