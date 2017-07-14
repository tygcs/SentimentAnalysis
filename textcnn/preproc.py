"""
pre processing
"""

import os
import chardet
import shutil
import random


def to_utf8(path):
    print('change file encoding to utf-8')
    encoding = 'utf-8'
    if os.path.isdir(path):
        for p in os.listdir(path):
            t = os.path.join(path, p)
            if os.path.isdir(t):
                to_utf8(t)
            else:
                change_encoding(t, encoding)
    else:
        change_encoding(path, encoding)


def change_encoding(filename, encoding):
    with open(filename, 'rb') as f:
        data = f.read()
    org_encoding = chardet.detect(data)['encoding']
    if org_encoding != encoding:
        os.remove(filename)
        org_encoding = 'gbk'
        try:
            data = data.decode(org_encoding).encode(encoding)
            with open(filename, 'wb') as f:
                f.write(data)
        except:
            pass


def build_dataset(data_path, force=True, proportion=0.2):
    print('build dataset')
    data_name = os.path.basename(data_path)
    dataset_path = os.path.join('temp', data_name)
    if not force and os.path.exists(dataset_path):
        return

    shutil.rmtree(dataset_path, ignore_errors=True)
    os.mkdir(dataset_path)

    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    os.mkdir(train_path)
    os.mkdir(test_path)

    pos_path = os.path.join(data_path, 'pos')
    neg_path = os.path.join(data_path, 'neg')
    pos_file = os.listdir(pos_path)
    neg_file = os.listdir(neg_path)

    random.shuffle(pos_file)
    random.shuffle(neg_file)
    test_pos_num = int(len(pos_file) * proportion)
    test_neg_num = int(len(neg_file) * proportion)

    test_pos_path = os.path.join(test_path, 'pos')
    test_neg_path = os.path.join(test_path, 'neg')
    os.mkdir(test_pos_path)
    os.mkdir(test_neg_path)
    for f in pos_file[:test_pos_num]:
        shutil.copy(os.path.join(pos_path, f), os.path.join(test_pos_path, f))
    for f in neg_file[:test_neg_num]:
        shutil.copy(os.path.join(neg_path, f), os.path.join(test_neg_path, f))

    train_pos_path = os.path.join(train_path, 'pos')
    train_neg_path = os.path.join(train_path, 'neg')
    os.mkdir(train_pos_path)
    os.mkdir(train_neg_path)
    for f in pos_file[test_pos_num:]:
        shutil.copy(os.path.join(pos_path, f), os.path.join(train_pos_path, f))
    for f in neg_file[test_neg_num:]:
        shutil.copy(os.path.join(neg_path, f), os.path.join(train_neg_path, f))


def parse_content(file):
    content = []
    with open(file, 'r', encoding='utf-8') as f:
        cnt = 0
        for line in f:
            cnt += 1
            if cnt % 6 == 5:
                l = line[9:line.find('</content>')]
                if l != '':
                    content.append(l)
    with open(os.path.join('data', 'sougouca'), 'w', encoding='utf-8') as f:
        for l in content:
            f.write(l + '\n')


def process(path):
    #to_utf8(path)
    build_dataset(path)


if __name__ == '__main__':
    data_path = r'data\htl_10000'
    process(data_path)
