import os
import sys
import json
import shutil

if __name__ == '__main__':
    with open('/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/oscar/datasets/custom_sets/custom.id_dictionary.txt') as f:
        content = f.readlines()
    id_dict = {}

    for line in content:
       value, key = line.split()
       id_dict[key] = value

    with open('/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/checkpoints/checkpoint-finetune-7-5/pred.coco_caption.custom.beam5.max20.odlabels.tsv') as f:
        content = f.readlines()
    results = {}
    for line in content:
        key, rest = line.split('\t')
        results[key] = json.loads(rest)[0]['caption']
    # print(results['3'])

    dataset_path = '/storage/plzen1/home/zeleznyt/DP/dataset/custom'
    dataset_root = os.path.abspath(os.path.join(dataset_path, os.pardir))
    dataset_file = dataset_path.split('/')[-1]
    renamed_dataset_path = os.path.join(dataset_root, dataset_file+'_renamed')
    # print(dataset_path)
    # print(dataset_root)
    # print(dataset_file)
    # print(os.path.join(dataset_root, dataset_file+'_renamed'))
    if not os.path.isdir(renamed_dataset_path):
        os.mkdir(renamed_dataset_path)
    # print(os.path.isdir(renamed_dataset_path))

    # input_path = os.path.join(sys.argv[1], 'train2017')
    image_list = [f for f in os.listdir(dataset_path)]
    for idx, item in enumerate(image_list):
        print(item)
        if id_dict[item] in results.keys():
            shutil.copy(os.path.join(dataset_path, item), os.path.join(renamed_dataset_path, item))
            os.rename(os.path.join(renamed_dataset_path, item),
                      os.path.join(renamed_dataset_path, results[id_dict[item]] + '.' + item.split('.')[-1]))
        else:
            shutil.copy(os.path.join(dataset_path, item), os.path.join(renamed_dataset_path, item))
            os.rename(os.path.join(renamed_dataset_path, item), os.path.join(renamed_dataset_path, id_dict[item]+'.'+item.split('.')[-1]))
