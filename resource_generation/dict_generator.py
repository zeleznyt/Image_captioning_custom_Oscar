import os
import sys


if __name__ == '__main__':
    tmp_idx = 1
    with open('id_dictionary2014.txt', 'w') as f:
        # input_path = os.path.join(sys.argv[1], 'train2017')
        # image_list = [os.path.join(input_path, f) for f in os.listdir(input_path)]
        # for idx, item in enumerate(image_list):
        #     id = item.split('/')[-1].split('.')[0]
        #     if id.isnumeric():
        #         f.write('{}\t{}\n'.format(tmp_idx, id))
        #         tmp_idx = tmp_idx +1
        #     else:
        #         print(id)
        #         print('ERROR')
        #         break
        # line_train = tmp_idx
        # print('train done on {} idx'.format(tmp_idx))

        input_path = os.path.join(sys.argv[1], 'val2014')
        image_list = [os.path.join(input_path, f) for f in os.listdir(input_path)]
        for idx, item in enumerate(image_list):
            id = item.split('/')[-1].split('.')[0]
            id = id.split('_')[-1]
            # print(id)
            if id.isnumeric():
                f.write('{}\t{}\n'.format(tmp_idx, id))
                tmp_idx = tmp_idx + 1
            else:
                print(id)
                print('ERROR')
                break
        line_val = tmp_idx
        print('val done on {} idx'.format(tmp_idx))

        input_path = os.path.join(sys.argv[1], 'test2014')
        image_list = [os.path.join(input_path, f) for f in os.listdir(input_path)]
        for idx, item in enumerate(image_list):
            id = item.split('/')[-1].split('.')[0]
            id = id.split('_')[-1]
            if id.isnumeric():
                f.write('{}\t{}\n'.format(tmp_idx, id))
                tmp_idx = tmp_idx + 1
            else:
                print(id)
                print('ERROR')
                break
        line_test = tmp_idx
        print('test done on {} idx'.format(tmp_idx))

    with open('id_dictionary2014.lineidx', 'w') as f:
        # f.write('{}\ttrain\n'.format(line_train))
        f.write('{}\tval\n'.format(line_val))
        f.write('{}\ttest\n'.format(line_test))


