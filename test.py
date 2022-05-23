#以下はサイトからのコピペである。

from spleeter.separator import Separator
import os

# 出力結果の保存場所をあらかじめ作っておく
for i in (2, 4, 5):  # 2音源、4音源、5音源
    outdir_path = './output/' + str(i) + 'stems'
    os.makedirs(outdir_path, exist_ok=True)

# 分離対象となる音楽wav
# https://soundcloud.com/ballforest/sample
input_audio = "./input/sample.wav"

# 初回実行時はモデルをダウンロードするため、「待ち」の時間がかかる
# 事前にダウンロードすることも可能 (pretrained_model/2stems などに保存)

# ボーカルとそれ以外に分離する(2音源)
separator_2stem = Separator('spleeter:2stems')
separator_2stem.separate_to_file(input_audio, "./output/2stems")

# ボーカル、ベース、ドラムとそれ以外に分離する(4音源)
separator_4stem = Separator('spleeter:4stems')
separator_4stem.separate_to_file(input_audio, "./output/4stems")

# ボーカル、ピアノ、ベース、ドラムとそれ以外に分離する(5音源)
separator_5stem = Separator('spleeter:5stems')
separator_5stem.separate_to_file(input_audio, "./output/5stems")