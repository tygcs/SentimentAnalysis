"""
project test
"""

import preproc
import wordseg
import tfword2vec
import classify
import gensim


def main():
    data_path = r'data\htl_2000'
    train_pos_path = r'C:\Users\neo\Desktop\chinese_hotel_comments\ChnSentiCorp_htl_ba_2000\pos\train'
    train_neg_path = r'C:\Users\neo\Desktop\chinese_hotel_comments\ChnSentiCorp_htl_ba_2000\neg\train'
    test_pos_path = r'C:\Users\neo\Desktop\chinese_hotel_comments\ChnSentiCorp_htl_ba_2000\pos\test'
    test_neg_path = r'C:\Users\neo\Desktop\chinese_hotel_comments\ChnSentiCorp_htl_ba_2000\neg\test'
    train_pos = 'train_pos_seg'
    train_neg = 'train_neg_seg'
    test_pos = 'test_pos_seg'
    test_neg = 'test_neg_seg'
    seg_file = 'seg'
    preproc.process(data_path)
    train_pos_lst, train_neg_lst = wordseg.pos_neg_segment(train_pos_path, train_pos, train_neg_path, train_neg)
    test_pos_lst, test_neg_lst = wordseg.pos_neg_segment(test_pos_path, test_pos, test_neg_path, test_neg)
    #wordseg.save_seg(train_pos_lst + train_neg_lst, seg_file)
    embedding_size = 100
    #tfwv = tfword2vec.TFWord2Vec(seg_file, embedding_size, 12000, 2, 4, None, 200, 100000)
    #wordvec = tfwv.word2vec('wordvec_comment', True)
    #wordvec = tfwv.word2vec('wordvec')
    wordvec = gensim.models.KeyedVectors.load_word2vec_format(r'divided_content.bin', binary=True)
    classify.do_classify(wordvec, embedding_size, train_pos_lst, train_neg_lst, test_pos_lst, test_neg_lst)


if __name__ == '__main__':
    main()
