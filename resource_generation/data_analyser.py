import json
from statistics import mean
# import matplotlib.pyplot as plt


def get_confidence_values(path, log_print=True, hist=False):
    with open(path, 'r') as f:
        content = f.readlines()

    # tmp = 0
    confs = []
    line_lens = []
    low_conf_images = []
    min_conf_values = []
    number_of_detections = []
    for line in content:
        line_confs = []
        # print(tmp)
        # tmp = tmp + len(line)
        line_list = json.loads(line.split('\t')[1])
        line_lens.append(len(line_list))
        number_of_detections.append(len(line_list))
        for item in line_list:
            confs.append(item['conf'])
            line_confs.append(item['conf'])

        if number_of_detections[-1] > 90:
            # print(confs)
            print(line.split('\t')[0])

        # print(line_confs)
        # if sum([v < 0.2 for v in line_confs]) > 0:
        #     # print(line_confs)
        #     # print(len(line_confs))
        #     if not len(line_confs) == 10:
        #         print('FATAL')
        #         break
        #     low_conf_images = low_conf_images + line_confs
        # if min(line_confs) > 0.4:
        #     print(line)
        #     for item in line_list:
        #         print(item['class'])
        min_conf_values.append(min(line_confs))
    if log_print:
        print('len : {}'.format(len(confs)))
        print('line len : {}'.format(len(line_lens)))
        # print('min : {}'.format(min(confs)))
        # print('max : {}'.format(max(confs)))
        print('max number of detetions: {}'.format(max(number_of_detections)))
        print('min number of detetions: {}'.format(min(number_of_detections)))
        print('mean of number of detetions: {}'.format(mean(number_of_detections)))
    # if hist:
    #     # plt.hist(x=low_conf_images, alpha=0.5, density=True, bins=100)
    #     plt.plot(min_conf_values)
    return confs


def print_label_confidence_info():
    # get_confidence_values(path='/home/tomas/fav/dp/output_features_from_detectron/test.label.tsv', hist=True)
    get_confidence_values(path='/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/oscar/datasets/provided-coco/coco_caption/train.label.tsv', hist=False)
    get_confidence_values(path='/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/oscar/datasets/provided-coco/coco_caption/test.label.tsv', hist=False)
    get_confidence_values(path='/storage/plzen1/home/zeleznyt/DP/oscar/Oscar/oscar/datasets/provided-coco/coco_caption/val.label.tsv', hist=False)
    # plt.legend(['train', 'test', 'val'])
    # plt.show()


if __name__ == "__main__":
    print_label_confidence_info()

