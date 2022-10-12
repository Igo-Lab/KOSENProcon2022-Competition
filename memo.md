# プログラムを作っていく上でのメモ

## 問題データの最大フレーム数
6963728フレーム．
問題データはこれを超えてくることはない．


## 分割データの最低フレーム数
24000フレーム

## len(問題データ)<len(元データ)
```
problem_data=[0, 1, 2, 3, 4]
data=[6, 7, 8, 9, 10, 11, 12, 13]


①～④過程
longest=max(problem_data.__len__(), data.__len__())
for j in range(1, longest+1, 1):
    clip_starti = max(0, j-data.__len__())
    data_starti = max(data.__len__()-j, 0)
    clipped=np.arange(problem_data.__len__(),dtype=np.int16)
    print(j)
    print(clipped[clip_starti:j])
    print(data[max(data.__len__()-j, 0):min(data.__len__(), data.__len__()+problem_data.__len__()-j)])

⑤～⑥過程
for j in range(max(problem_data.__len__()+1, data.__len__()+1), problem_data.__len__()+data.__len__(), 1):
    clipped=np.arange(problem_data.__len__(),dtype=np.int16)
    print(j)
    print(clipped[min(j-problem_data.__len__()-data.__len__(), 0):clipped.__len__()])
    print(data[0:min(problem_data.__len__(),problem_data.__len__()+data.__len__()-j)])
```

## len(問題データ)>len(元データ)
```
problem_data=[0, 1, 2, 3, 4, 5, 6, 7]
data=[7, 8, 9, 10, 11]


①～③過程
for j in range(1, max(problem_data.__len__()+1, data.__len__()+1), 1):
    clipped=np.arange(problem_data.__len__(),dtype=np.int16)
    print(j)
    print(clipped[max(0, j-data.__len__()):j])
    print(data[max(data.__len__()-j, 0):min(data.__len__(), data.__len__()+problem_data.__len__()-j)])

⑤～⑥
for j in range(max(problem_data.__len__()+1, data.__len__()+1), problem_data.__len__()+data.__len__(), 1):
    clipped=np.arange(problem_data.__len__(),dtype=np.int16)
    print(j)
    print(clipped[min(j-problem_data.__len__()-data.__len__(), 0):clipped.__len__()])
    print(data[0:min(problem_data.__len__(),problem_data.__len__()+data.__len__()-j)])
```


## 全てを統合
```py

for j in range(1, problem_data.__len__()+data.__len__(), 1):
    clip_starti = max(0, j-data.__len__())
    clip_endi = min(j, problem_data.__len__())
    data_starti = max(data.__len__()-j, 0)
    data_endi = min(data.__len__(), data.__len__()+problem_data.__len__()-j)
    clipped=np.zeros((problem_data.__len__(),),dtype=np.int16)
    print(j, clip_starti, clip_endi, data_starti, data_endi)
    #clipped[clip_starti:j]=data[data_starti:data_endi]

    subbed = (
        problem_data[clip_starti:clip_endi] - data[data_starti:data_endi]
    )

    remaining = (
                    np.sum(np.abs(subbed))
                    + np.sum(np.abs(problem_data[0:clip_starti]))
                    + np.sum(np.abs(problem_data[clip_endi : len(problem_data)]))
                )
    print(remaining)
    print(clipped)
```

# 外れ値検出
進捗メモ
外れ値検出に候補ができた。面積を昇順で並び替えたあと、小さい順から重ね合わせ数だけ取り除いたあと、残りの配列の平均値・標準偏差を計算して、それらを使って取り除いておいた要素も標準化する。


# threads per smについて
maxwellの場合2048
2048/256=8threads/sm
2048/32=64warps/sm
8block/sm

# ライブラリのインストールについて
　Python3.10.7入れた．

# サンプリングレート
　48kHz

# Jetsonのセットアップについて
セットアップ後，スクリーンセーバー無効化・アップデートの確認無効化・ファンの自動回転スクリプト（下記）を/etc/rc.locaに書き込む．rc.localにchmod +xしないといけない．
```
sudo sh -c 'echo 80 > /sys/devices/pwm-fan/target_pwm'
```

# 並び替えアルゴリズム

```python
thebin = []
for idx in random:
    thebin.append(idx)
    cont = [thebin[-1]]
    head = thebin[-1]
    tail = thebin[-1]
    while True:
        if head-1 in thebin:
            cont.insert(0, head-1)
            head -= 1
        elif tail+1 in thebin:
            cont.append(tail+1)
            tail += 1
        else:
            break
    print(cont)
```