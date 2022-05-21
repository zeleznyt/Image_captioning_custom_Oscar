import os
import sys
import json

if __name__ == '__main__':
    # with open('captions_val2014.json', 'r') as f:
    with open('/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/oscar/datasets/coco_caption_no-outliers/captions_val2014.json', 'r') as f:
        content = f.readlines()
    content_list = json.loads(content[0])
    print((content_list.keys()))
    print(content_list['info'])
    print(content_list['licenses'])
    print(content_list['images'][0])
    print(content_list['annotations'][0])
    with open('id_dictionary.txt', 'r') as f:
    # with open('/storage/plzen1/home/zeleznyt/DP/id_dictionary.txt', 'r') as f:
        content = f.readlines()

    id_dict = {}
    for line in content:
        value, key = line.split()
        id_dict[key] = value

    outliers = []
    with open('new.val2014.outliers.txt', 'r') as f:
    # with open('/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/oscar/datasets/coco_caption/new.train2017.outliers.txt', 'r') as f:
        content = f.readlines()
    for item in content:
        outliers.append(int(item))

    viables = []
    with open('new.val2014.label.tsv',
    # with open('/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/oscar/datasets/coco_caption/new.train2017.label.tsv',
              'r') as f:
        content = f.readlines()
    for item in content:
        viables.append(int(item.split()[0]))

    result = {'annotations': [], 'images': [], 'type': 'captions', 'info': 'dummy', 'licenses': 'dummy'}
    # result_annotations = []

    for anot in content_list['annotations']:
        new_anot = anot.copy()
        # print(anot['image_id'])
        # print(type(anot['image_id']))
        # print(id_dict[str(anot['image_id']).zfill(12)])
        new_anot['image_id'] = str(id_dict[str(anot['image_id']).zfill(12)])
        if int(new_anot['image_id']) in outliers:
            # print('{} skipped'.format(new_anot['image_id']))
            continue
        if int(new_anot['image_id']) in viables:
            # print(anot)
            # print(new_anot)
            result['annotations'].append(new_anot)
            # print(new_anot['image_id'])
            result['images'].append({'id': new_anot['image_id'], 'file_name': new_anot['image_id']})

    # for annot in content_list['annotations']:
    #     res_annot = {annot['']}
    #     break
    #     result['annotations'].append(res_annot)
    # with open('new.val2017_caption_coco_format.json', 'w') as f:
    #     f.write(json.dumps(content_list['annotations']))

    with open('val2014_caption_coco_format.json', 'w') as f:
        f.write(json.dumps(result))
