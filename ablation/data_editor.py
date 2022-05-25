import ast
import argparse
import base64
import os
import json
import numpy as np
import random
import copy


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-prefix",
        default="ablaset."
    )
    parser.add_argument(
        "--new-prefix",
        default="edited.ablaset"
    )
    parser.add_argument(
        "--input-path",
        default="ablaset_features",
        help="path to file with input images",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="A directory to save output features."
    )
    parser.add_argument(
        "--coco-classnames",
        default="ms_coco_classnames.txt",
        help="File with coco classes names"
    )
    parser.add_argument(
        "--edited-images-ids",
        default="edited_images_ids.txt"
    )
    parser.add_argument(
        "--modification",
        default="",
        help="One of following ones: drop_major, null_tags, null_feats, null_feats_change_class"
    )
    return parser


def load_data(feature_file, label_file):
    with open(label_file, 'r') as f:
        label_content = f.readlines()
    data_info = {}
    for i, line in enumerate(label_content):
        id = line.split('\t')[0]
        data_info[id] = {'labels': json.loads(line.split('\t')[1])}

    with open(feature_file, 'r') as f:
        feature_content = f.readlines()
    for i, line in enumerate(feature_content):
        id = line.split('\t')[0]
        feat_dict = json.loads(line.split('\t')[1])
        featsb = base64.b64decode(feat_dict['features'])
        num_boxes = int(feat_dict['num_boxes'])
        feats = np.frombuffer(featsb, dtype=np.float64)

        data_info[id]['num_boxes'] = num_boxes
        data_info[id]['features'] = feats
    return data_info


def prep_files(data_info=None):
    features_file = os.path.join(args.output_path, '{}feature.tsv'.format(args.new_prefix))
    labels_file = os.path.join(args.output_path, '{}label.tsv'.format(args.new_prefix))

    with open(features_file, 'w') as f:
        pass
    with open(labels_file, 'w') as f:
        pass
    with open(os.path.join(args.output_path, '{}feature.lineidx'.format(args.new_prefix)), 'w') as f:
        pass
    with open(os.path.join(args.output_path, '{}label.lineidx'.format(args.new_prefix)), 'w') as f:
        pass
    with open(os.path.join(args.output_path, '{}yaml'.format(args.new_prefix)), 'w') as f:
        f.write('img: img.tsv\nhw: hw.tsv\nlabel: {}label.tsv\nfeature: {}feature.tsv'.format(args.new_prefix,
                                                                                              args.new_prefix))
    with open(os.path.join(args.output_path, '{}drop_outliers.txt'.format(args.new_prefix)), 'w') as f:
        pass
    if data_info is not None:
        len_feature = 0
        len_label = 0
        for image_id in data_info.keys():
            with open(os.path.join(args.output_path, '{}feature.lineidx'.format(args.new_prefix)), 'a') as f:
                f.write('{}\n'.format(len_feature))
            with open(os.path.join(args.output_path, '{}label.lineidx'.format(args.new_prefix)), 'a') as f:
                f.write('{}\n'.format(len_label))

            features_array = data_info[image_id]['features']
            sb = base64.b64encode(features_array)
            s = sb.decode("utf-8")
            features_info = {"num_boxes": data_info[image_id]['num_boxes'], "features": s}
            labels_info = data_info[image_id]['labels']

            with open(features_file, 'a') as f:
                text = '{}\t{}\n'.format(image_id, json.dumps(features_info))
                len_feature = len_feature + len(text)
                f.write(text)
            with open(labels_file, 'a') as f:
                text = '{}\t{}\n'.format(image_id, json.dumps(labels_info))
                len_label = len_label + len(text)
                f.write(text)
        return len_feature, len_label


def save_data(data_info, len_feature=0, len_label=0):
    features_file = os.path.join(args.output_path, '{}feature.tsv'.format(args.new_prefix))
    labels_file = os.path.join(args.output_path, '{}label.tsv'.format(args.new_prefix))

    for image_id in data_info.keys():
        with open(os.path.join(args.output_path, '{}feature.lineidx'.format(args.new_prefix)), 'a') as f:
            f.write('{}\n'.format(len_feature))
        with open(os.path.join(args.output_path, '{}label.lineidx'.format(args.new_prefix)), 'a') as f:
            f.write('{}\n'.format(len_label))

        features_array = data_info[image_id]['features']
        sb = base64.b64encode(features_array)
        s = sb.decode("utf-8")
        features_info = {"num_boxes": data_info[image_id]['num_boxes'], "features": s}
        labels_info = data_info[image_id]['labels']

        with open(features_file, 'a') as f:
            text = '{}\t{}\n'.format(image_id, json.dumps(features_info))
            len_feature = len_feature + len(text)
            f.write(text)
        with open(labels_file, 'a') as f:
            text = '{}\t{}\n'.format(image_id, json.dumps(labels_info))
            len_label = len_label + len(text)
            f.write(text)


def edit_data(_data, modification):
    new_data = copy.deepcopy(_data)
    for im_idx, im_id in enumerate(_data.keys()):
        im = new_data[im_id]
        if modification == 'drop_major':
            dropped_times = 0
            major_class = im['labels'][0]['class']
            new_im = {'labels': [], 'num_boxes': 0, 'features': np.array([])}
            for idx, detection in enumerate(im['labels']):
                if detection['class'] == major_class:
                    dropped_times += 1
                    continue
                else:
                    new_im['labels'].append(detection)
                    new_im['num_boxes'] += 1
                    oneFeatLen = len(im['features'])//im['num_boxes']
                    if len(new_im['features']) > 0:
                        new_im['features'] = np.concatenate((new_im['features'], im['features'][idx*oneFeatLen: (idx+1)*oneFeatLen]))
                    else:
                        new_im['features'] = im['features'][idx*oneFeatLen: (idx+1)*oneFeatLen]

            if new_im['num_boxes'] < 1:
                print('{} can not drop major class {} (it is the only one)'.format(im_id, major_class))
                with open(os.path.join(args.output_path, '{}drop_outliers.txt'.format(args.new_prefix)), 'a') as f:
                    f.write('{}\n'.format(im_id))
                continue
            else:
                new_data[im_id] = new_im
                mode2s = 'Dropped {}: {}x'.format(major_class, dropped_times)
        elif modification == 'null_tags':
            for detection in im['labels']:
                detection['class'] = ''
                # detection['rect'] = [0, 0, 0, 0]
                # detection['conf'] = 0
            mode2s = 'All tags nulled'
        elif modification == 'null_feats':
            new_feats = np.zeros(shape=np.shape(im['features']), dtype=float)
            im['features'] = new_feats
            mode2s = 'All feats nulled'
        elif modification == 'null_feats_change_class':
            new_feats = np.zeros(shape=np.shape(im['features']), dtype=float)
            im['features'] = new_feats
            with open(args.coco_classnames, 'r') as f:
                content = f.read()
            coco_classnames = ast.literal_eval(content)
            new_classes = []
            for detection in im['labels']:
                random_class = random.randint(1, len(coco_classnames) - 1)
                detection['class'] = coco_classnames[random_class]
                new_classes.append(detection['class'])
            mode2s = 'All feats nulled, classes added: {}'.format(new_classes)
        else:
            return -1

        with open(args.edited_images_ids, 'a') as f:
            info = {'original_id': im_id, 'settings': modification, 'mode2s': mode2s}
            f.write('{}\n'.format(json.dumps(info)))
    return new_data


if __name__ == '__main__':
    random.seed(1)
    args = get_parser().parse_args()
    print(args)
    label_file = os.path.join(args.input_path, '{}label.tsv'.format(args.old_prefix))
    feature_file = os.path.join(args.input_path, '{}feature.tsv'.format(args.old_prefix))

    data = load_data(label_file=label_file, feature_file=feature_file)
    prep_files()
    print('Files prepared in: {}'.format(args.output_path))

    result_data = dict(data)
    max_idx = max([int(x) for x in result_data])
    new_idx = -1
    with open(args.edited_images_ids, 'w') as f:
        pass

    modifications = ['drop_major', 'null_tags', 'null_feats', 'null_feats_change_class']
    if args.modification not in modifications:
        print('Modification {} no found. Try on of following ones: drop_major, null_tags, null_feats, null_feats_change_class'.format(args.modification))
    mode = args.modification

    edited_data = edit_data(data, modification=mode)
    if edited_data != -1:
        save_data(edited_data)
        print('Edited data saved to {}'.format(args.output_path))
    else:
        print('No data was created')
