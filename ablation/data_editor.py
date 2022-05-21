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
        default='ablaset.'
    )
    parser.add_argument(
        "--new-prefix",
        default='edited.ablaset.'
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
    # parser.add_argument(
    #     "--edit-rect",
    #     default=0,
    #     # help="0 - no edit, 1 - adding noise, 2 - totally random"
    # )
    # parser.add_argument(
    #     "--edit-feat",
    #     default=0,
    #     # help="0 - no edit, 1 - adding noise, 2 - totally random"
    # )
    # parser.add_argument(
    #     "--edit-class",
    #     default=0,
    #     help="0 - no edit, >0 - random"
    # )
    # parser.add_argument(
    #     "--edit-conf",
    #     default=0,
    #     # help="0 - no edit, 1 - adding noise, 2 - totally random"
    # )
    parser.add_argument(
        "--edited-images-ids",
        default='edited_images_ids.txt',
        # help="0 - no edit, 1 - adding noise, 2 - totally random"
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
        # feats = feats.reshape(num_boxes, -1)

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


# def edit_data(_data, edit_rect=0, edit_feat=0, edit_class=0, edit_conf=0, start_idx=0):
def edit_data(_data, modification, start_idx=0):
    new_data = copy.deepcopy(_data)
    new_idx = start_idx
    for im_idx, im_id in enumerate(_data.keys()):
        im = new_data[im_id]
        if modification == 'drop_some':
            n = 1
            dropped = []
            for i in range(n):
                if i > len(im['labels']):
                    print('Image {} has not enough detections: {}'.format(im_id, n))
                    break
                dropped.append(im['labels'][0]['class'])
                im['features'] = im['features'][len(im['features'])//im['num_boxes']:]
                im['labels'] = im['labels'][1:]
                im['num_boxes'] -= 1
            mode2s = 'Dropped {}:{}'.format(n, dropped)
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

        # if int(edit_rect) > 0:
        #     pass
        # if int(edit_feat) > 0:
        #     old_feats = im['features']
        #     new_feats = np.empty(shape=np.shape(im['features']), dtype=float)
        #     for i in range(len(new_feats)):
        #         new_feats[i] = 0#abs(old_feats[i]+1e-100)#random.uniform(-old_feats[i]/10, old_feats[i]/10))
        #     # print('Features edited')
        #     # print('old features shape: {}'.format(np.shape(old_feats)))
        #     # print(type(old_feats))
        #     # print('new features shape: {}'.format(np.shape(new_feats)))
        #     # print(type(new_feats))
        #     im['features'] = new_feats
        #
        # if int(edit_class) > 0:
        #     with open(args.coco_classnames, 'r') as f:
        #         content = f.read()
        #     coco_classnames = ast.literal_eval(content)
        #     for detection in im['labels']:
        #         random_class = random.randint(1, len(coco_classnames) - 1)
        #         detection['class'] = coco_classnames[random_class]
        # if int(edit_conf) > 0:
        #     for detection in im['labels']:
        #         random_conf = random.uniform(0, 1)
        #         detection['conf'] = random_conf
        new_data[str(im_idx + new_idx)] = new_data.pop(im_id)
        with open(args.edited_images_ids, 'a') as f:
            info = {'original_id': im_id, 'new_id': str(im_idx + new_idx), 'settings': modification, 'mode2s': mode2s}
            f.write('{}\n'.format(json.dumps(info)))
        new_idx += 1
    return new_data, new_idx


if __name__ == '__main__':
    random.seed(1)
    args = get_parser().parse_args()
    print(args)
    label_file = os.path.join(args.input_path, '{}label.tsv'.format(args.old_prefix))
    feature_file = os.path.join(args.input_path, '{}feature.tsv'.format(args.old_prefix))

    data = load_data(label_file=label_file, feature_file=feature_file)
    prep_files(data)
    print('Files prepared in: {}'.format(args.output_path))

    # edits_rect = [0]
    # edits_feat = [0, 1]
    # edits_class = [0,1]
    # edits_conf = [0]
    result_data = dict(data)
    max_idx = max([int(x) for x in result_data])
    new_idx = -1
    with open(args.edited_images_ids, 'w') as f:
        pass
    # modifications = ['drop_some', 'null_tags', 'null_feats', 'null_feats_change_class']
    modifications = ['null_tags']
    for mode in modifications:
        if new_idx < 0:
            edited_data, new_idx = edit_data(data, modification=mode, start_idx=max_idx + 1)
        else:
            edited_data, new_idx = edit_data(data, modification=mode, start_idx=new_idx + 1)
        if edited_data == -1:
            continue
        save_data(edited_data)
        # result_data.update(edited_data)
    # for edit_rect in edits_rect:
    #     for edit_feat in edits_feat:
    #         for edit_class in edits_class:
    #             for edit_conf in edits_conf:
    #                 if np.sum([edit_rect, edit_feat, edit_class, edit_conf]) == 0:
    #                     print('Skipping setting with no edit')
    #                     continue
    #                 print('Editing for rect: {},  Feat: {}, Class: {}, Conf: {}'.format(edit_rect, edit_feat,
    #                                                                                     edit_class, edit_conf))
    #                 edited_data = edit_data(data, edit_rect=edit_rect, edit_feat=edit_feat, edit_class=edit_class,
    #                                         edit_conf=edit_conf, start_idx=max([int(x) for x in result_data]) + 1)
    #
    #                 result_data.update(edited_data)

    # print(data['0'])
    # print(len(data.keys()))
    #
    # print(result_data['0'])
    # print(len(result_data.keys()))

    # save_data(result_data)
    print('Edited data saved to {}'.format(args.output_path))
