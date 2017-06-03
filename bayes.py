# -*- coding: utf-8 -*-
import time
import jieba
from process_data import load_corpus_data, read_hotel_cmts


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def all_chinese(text):
    for w in text:
        if not is_chinese(w):
            return False
    return True


def compute_text(text, sentiment_words, train_p_pos, train_p_neg, pos_words_cnt, neg_words_cnt):

    pos_p = 1
    neg_p = 1

    cut_list = jieba.cut(text)
    for word in cut_list:
        if not train_p_pos.has_key(word) and not train_p_neg.has_key(word):
            continue
        if not train_p_neg.has_key(word) and train_p_pos.has_key(word):
            train_p_neg[word] = 1
            neg_words_cnt += 1
        if not train_p_pos.has_key(word) and train_p_neg.has_key(word):
            train_p_pos[word] = 1
            pos_words_cnt += 1
        pos_p = pos_p * train_p_pos[word] * 1.0/pos_words_cnt
        neg_p = neg_p * train_p_neg[word] * 1.0/neg_words_cnt

    return pos_p*1.0/(pos_p+neg_p)


def train_bayes():

    t1 = time.time()
    print "load corpus data!"
    pos_words, neg_words, judge_words, double_judge_words, degree_words, stopwords = load_corpus_data()

    print "merge sentiment words!"
    sentiment_words = []
    sentiment_words.extend(pos_words)
    sentiment_words.extend(neg_words)

    p_pos = {}
    p_neg = {}
    pos_words_cnt = 0
    neg_words_cnt = 0

    print "merge sentiment words done! ", time.time()-t1

    for i in range(1600):
        try:
            text = read_hotel_cmts("pos", i)
            cut_text = jieba.cut(text)
            for word in cut_text:
                if all_chinese(word):  # and word not in stopwords:
                    if p_pos.has_key(word):
                        p_pos[word] += 1
                    else:
                        p_pos[word] = 2
                        pos_words_cnt += 1
                    pos_words_cnt += 1
        except:
            pass

    print "train pos comments done! ", time.time()-t1

    for i in range(1600):
        try:
            text = read_hotel_cmts("neg", i)
            cut_text = jieba.cut(text)
            for word in cut_text:
                if all_chinese(word):  # and word not in stopwords:
                    if p_neg.has_key(word):
                        p_neg[word] += 1
                    else:
                        p_neg[word] = 2
                        neg_words_cnt += 1
                    neg_words_cnt += 1
        except:
            pass

    print "train neg comments done! ", time.time()-t1

    return p_pos, p_neg, pos_words_cnt, neg_words_cnt, sentiment_words


def test_hotel_comments():
    p_pos, p_neg, pos_words_cnt, neg_words_cnt, sentiment_words = train_bayes()
    pos_cnt = 0
    pos_all = 400
    neg_cnt = 0
    neg_all = 400

    print "begin test on hotel comments!"

    for i in range(1600, 2000):
        try:
            text = read_hotel_cmts("pos", i)
            score = compute_text(text, sentiment_words, p_pos, p_neg, pos_words_cnt, neg_words_cnt)
            if score > 0.5:
                pos_cnt += 1
        except:
            pos_all -= 1

    for i in range(1600, 2000):
        try:
            text = read_hotel_cmts("neg", i)
            score = compute_text(text, sentiment_words, p_pos, p_neg, pos_words_cnt, neg_words_cnt)
            if score <= 0.5:
                neg_cnt += 1
        except:
            neg_all -= 1

    print pos_cnt * 1.0 / pos_all, neg_cnt * 1.0 / neg_all, (pos_cnt + neg_cnt) * 1.0 / (pos_all + neg_all)
    print "testing done!"


def test():
    p_pos, p_neg, pos_words_cnt, neg_words_cnt, sentiment_words = train_bayes()
    text = u'位置离我们单位很近,从价格来说,性价比很高.我要的大床房,168元,前台服务员态度很好,房间硬件一般,但是想想价格也就这样了.还算干净,就是床垫子太硬.'
    print compute_text(text, sentiment_words, p_pos, p_neg, pos_words_cnt, neg_words_cnt)


if __name__ == '__main__':
    test_hotel_comments()
