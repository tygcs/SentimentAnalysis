import time
import jieba
from process_data import load_corpus_data, read_hotel_cmts


def test_result():

    # import dict

    pos_words, neg_words, judge_words, double_judge_words, degree_words, stopwords = load_corpus_data()

    for word in pos_words:
        jieba.add_word(word)
    for word in neg_words:
        jieba.add_word(word)
    for word in judge_words:
        jieba.add_word(word)
    for word in double_judge_words:
        jieba.add_word(word)

    # test text files

    pos_true = 0
    pos_all = 1000
    neg_true = 0
    neg_all = 1000

    for i in range(1000):
        try:
            text = read_hotel_cmts("pos", i)
            score = get_score(text, pos_words, neg_words, judge_words, double_judge_words, degree_words)
            if score > 0:
                pos_true += 1
        except:
            pos_all -= 1

    for i in range(1000):
        try:
            text = read_hotel_cmts("neg", i)
            score = get_score(text, pos_words, neg_words, judge_words, double_judge_words, degree_words)
            if score <= 0:
                neg_true += 1
        except:
            neg_all -= 1

    print pos_true, pos_all, neg_true, neg_all, pos_true*1.0/pos_all, neg_true*1.0/neg_all, (pos_true+neg_true)*1.0/(pos_all+neg_all)
    return pos_true, pos_all, neg_true, neg_all


def get_score(text, pos_words, neg_words, judge_words, double_judge_words, degree_words):

    cut_list = jieba.lcut(text)

    pos = []
    neg = []
    score = 0
    length = len(cut_list)
    for i in range(length):
        word = cut_list[i]
        if word in pos_words:
            if i > 0:
                if cut_list[i-1] in judge_words:
                    if i - 1 > 0 and ((cut_list[i-2] in judge_words) or
                                          (cut_list[i-2]+cut_list[i-1] in double_judge_words)):
                        score += 1
                    elif i - 1 > 0 and cut_list[i-2] in degree_words:
                        score -= 2
                    else:
                        score -= 1
                elif cut_list[i-1] in degree_words:
                    score += 3
                else:
                    score += 1
            else:
                score += 1
        elif word in neg_words:
            if i > 0:
                if cut_list[i-1] in judge_words:
                    if i - 1 > 0 and ((cut_list[i-2] in judge_words) or
                                          (cut_list[i-2]+cut_list[i-1] in double_judge_words)):
                        score -= 1
                    elif i - 1 > 0 and cut_list[i-2] in degree_words:
                        score += 1
                    else:
                        score += 1
                elif cut_list[i-1] in degree_words:
                    score -= 4
                else:
                    score -= 3
            else:
                score -= 2

    # map the score to [-1, 0, 1]

    sentiment = 0
    if score > 0:
        sentiment = 1
    elif score < 0:
        sentiment = -1

    return sentiment


if __name__ == '__main__':
    t1 = time.time()
    test_result()
    t2 = time.time()
    print "time: ", t2 - t1
