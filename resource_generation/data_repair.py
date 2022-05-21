import os
import sys


def repair_label_tsv():
    print('reading...')
    with open('/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/oscar/datasets/coco_caption/new.train2017.feature.lineidx', 'r') as f:
        content_lidx = f.readlines()
    print('lineidx loaded...')
    with open('/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/oscar/datasets/coco_caption/new.train2017.feature.tsv', 'r') as f:
        content_ltsv = f.readlines()
    result_ltsv = []
    result_lidx = [0]
    print('loading...')
    for i, line in enumerate(content_ltsv):
        if i < 85393:
            result_ltsv.append(line)
            continue
        idx = int(line.split()[0])
        idx = idx + 105984
        new_line = str(idx)+ line[1:]
        result_ltsv.append(new_line)
        # print(i, line)
        # print(new_line)
        # break

    print('writing...')
    with open('/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/oscar/datasets/coco_caption/new.train2017.feature.tsv', 'w') as f:
        for line in result_ltsv:
            f.write(line)
            result_lidx.append(result_lidx[-1]+len(line))

    with open('/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/oscar/datasets/coco_caption/new.train2017.feature.lineidx', 'w') as f:
        for i, line in enumerate(result_lidx):
            if i == len(result_lidx)-1:
                break
            f.write(str(line)+'\n')


if __name__ == '__main__':
    repair_label_tsv()
