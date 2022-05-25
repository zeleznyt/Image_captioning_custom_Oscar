import argparse
import os
import json


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edited-images-ids",
        default='edited_images_ids.txt'
    )
    parser.add_argument(
        "--id-dictionary",
        default='id_dictionary.txt'
    )
    parser.add_argument(
        "--pred-tsv",
        default='pred.coco_caption.edited.ablaset.beam5.max20.odlabels.tsv'
    )
    parser.add_argument(
        "--output-path",
        default=''
    )
    return parser


def load_files():
    with open(args.id_dictionary, 'r') as f:
        content = f.readlines()
    id_dict = {}
    for line in content:
        key, value = line.split()
        id_dict[key] = value

    with open(args.pred_tsv, 'r') as f:
        content = f.readlines()
    predictions = {}
    for line in content:
        key, rest = line.split('\t')
        predictions[key] = json.loads(rest)[0]['caption']

    with open(args.edited_images_ids, 'r') as f:
        content = f.readlines()
    edited_images_ids = {}
    for line in content:
        edited_images_ids[json.loads(line)['new_id']] = json.loads(line)

    return id_dict, predictions, edited_images_ids


if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args)
    id_dict, predictions, edited_images_ids = load_files()

    print(edited_images_ids)
    print(predictions)
    print(id_dict)

    # Prepairing the folder with files
    result_path = os.path.join(args.output_path, 'result_tsvs')
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    for image_id in id_dict.keys():
        with open(os.path.join(result_path, image_id.zfill(2)+'.tsv'), 'w') as f:
            f.write('{}\t{}\t{}\t{}\t{}\n'.format('new_id', 'file_name', 'coco_id', 'settings', 'prediction'))

    # Filling the files
    for pred in predictions.keys():
        if pred not in edited_images_ids:
            original_id = pred
            new_id = pred
            settings = 'no_modification'
            mode2s = ''
            file_name = id_dict[original_id]
            coco_id = file_name.split('.')[0].strip('0')
            prediction = predictions[pred]
        else:
            original_id = edited_images_ids[pred]['original_id']
            new_id = edited_images_ids[pred]['new_id']
            settings = edited_images_ids[pred]['settings']
            mode2s = edited_images_ids[pred]['mode2s']
            file_name = id_dict[original_id]
            coco_id = file_name.split('.')[0].strip('0')
            prediction = predictions[pred]
        with open(os.path.join(result_path, original_id.zfill(2)+'.tsv'), 'a') as f:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(new_id.zfill(2), file_name, coco_id.zfill(5), settings, prediction, mode2s))