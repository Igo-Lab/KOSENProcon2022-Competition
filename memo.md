# プログラムを作っていく上でのメモ

## 元データの最大フレーム数
394606フレーム．
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
