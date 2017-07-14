"""
word segmentation
"""

import jieba
import os
import re


def seg_data(data_path):
    print('seg_data', data_path)
    res = []
    if os.path.isdir(data_path):
        for file in os.listdir(data_path):
            print('segment', file)
            with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
                txt = f.read()
            # remove '\t' '\n'
            txt = re.sub('\s+', ' ', txt)
            if txt == '':
                continue
            l = jieba.lcut(txt)
            res.append(l)
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            cnt = 0
            for line in f:
                line = re.sub('\s+', ' ', line)
                if line == '':
                    continue
                l = jieba.lcut(line)
                res.append(l)
                cnt += 1
                if cnt % 5000 == 0:
                    print('segment %d lines' % cnt)
    print('seg_data end')
    return res


def rm_word(seg_lst, rm_word_file):
    print('rm_word', rm_word_file)
    with open(rm_word_file, 'r', encoding='utf-8') as f:
        rm_word = f.read()
        rm_word = rm_word.split('\n')
    cnt = 0
    for l in seg_lst:
        idx = len(l) - 1
        while idx >= 0:
            if l[idx] in rm_word:
                del l[idx]
            idx -= 1
        cnt += 1
        if cnt % 5000 == 0:
            print('process %d items' % cnt)
    seg_lst = [s for s in seg_lst if s != '']
    print('rm_word process %d items' % cnt)


def save_seg(seg_lst, seg_file):
    print('save_seg', seg_file)
    with open(seg_file, 'w', encoding='utf-8') as f:
        cnt = 0
        for s in seg_lst:
            f.write(s + '\n')
            cnt += 1
            if cnt % 5000 == 0:
                print('write %d lines' % cnt)
    print('save_seg write %d lines' % cnt)


def segment(data_path, seg_file, force=False):
    if not force and os.path.exists(seg_file):
        with open(seg_file, 'r', encoding='utf-8') as f:
            seg_lst = list(f.readlines())
            seg_lst = [s.strip() for s in seg_lst]
        return seg_lst

    seg_lst = seg_data(data_path)
    rm_word(seg_lst, os.path.join('data', 'punctuation.txt'))
    #rm_word(seg_lst, os.path.join('data', 'stopwords.txt'))
    seg_lst = [' '.join(l).lower() for l in seg_lst]
    save_seg(seg_lst, seg_file)
    return seg_lst


def seg(data_path, force=False):
    pos_data_path = os.path.join(data_path, 'pos')
    neg_data_path = os.path.join(data_path, 'neg')
    pos_seg_file = os.path.join(data_path, 'pos_seg')
    neg_seg_file = os.path.join(data_path, 'neg_seg')
    pos_seg_lst = segment(pos_data_path, pos_seg_file, force)
    neg_seg_lst = segment(neg_data_path, neg_seg_file, force)
    return pos_seg_lst, neg_seg_lst


if __name__ == '__main__':
    train_data_path = r'temp\htl_10000\train'
    test_data_path = r'temp\htl_10000\test'
    seg(train_data_path, True)
    seg(test_data_path, True)
    #sougouca_path = 'sougouca'
    #lst = seg_data(sougouca_path)
    #rm_word(lst, 'punctuation.txt')
    #save_seg(lst, 'sougouca_seg')

