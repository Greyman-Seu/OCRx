from params import alphabet
import pandas as pd

filename = 'log45.txt'
alphabet_hash = {char: i for i, char in enumerate(alphabet)}

preds = []
gts = []
with open(filename, 'r') as f:
    for line in f:
        pred = line[30:50].rstrip()
        gt = line[56:-1].rstrip()
        if gt:
            preds.append(pred)
            gts.append(gt)
preds = preds[:-1]
gts = gts[:-1]
diff_len = {i for i in range(len(gts)) if len(gts[i]) != len(preds[i])}

# 原始数据信息
data_raw = pd.DataFrame({'gts': gts, 'preds': preds})

maps = [{} for _ in alphabet]
for i, (gt, pred) in enumerate(zip(gts, preds)):
    if i in diff_len:
        continue
    for y, x in zip(gt, pred):
        ind = alphabet_hash[y]
        if x not in maps[ind]:
            maps[ind][x] = 1
        else:
            maps[ind][x] += 1

# 统计信息版
data_stat = pd.DataFrame(maps, index=list(alphabet), columns=list(alphabet))
diff = pd.DataFrame([(gts[i], preds[i])
                     for i in diff_len], columns=['gts', 'preds'])
