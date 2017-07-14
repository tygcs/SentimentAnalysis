"""
classification
"""

import collections
import math

import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def svm_classification(train_x, train_y, test_x):
    clf = svm.SVC(C=0.02, kernel='poly', degree=10, coef0=10, gamma=0.2)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    return predict_y


def bayes_classification(train_x, train_y, test_x):
    clf = GaussianNB()
    clf.fit(train_x, train_y)
    return clf.predict(test_x)


def ada_classification(train_x, train_y, test_x):
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(train_x, train_y)
    return clf.predict(test_x)


def rf_classification(train_x, train_y, test_x):
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(train_x, train_y)
    return clf.predict(test_x)


def kn_classification(train_x, train_y, test_x):
    clf = KNeighborsClassifier(10)
    clf.fit(train_x, train_y)
    return clf.predict(test_x)


def get_idf(seg_lst):
    df = {}
    for l in seg_lst:
        l = l.split()
        counter = collections.Counter(l)
        for word in counter:
            if word not in df:
                df[word] = 0
            df[word] += 1
    idf = {}
    doc_cnt = len(seg_lst)
    for word in df:
        cnt = df[word]
        idf[word] = math.log(doc_cnt / cnt)
    return idf


def build_data(wordvec, embedding_size, seg_lst, flag, idf):
    X = []
    Y = []
    for l in seg_lst:
        l = l.split()
        counter = collections.Counter(l)
        sum_tfidf = 0
        unk_cnt = 0
        vec_lst = []
        for word in counter:
            tf = counter[word]
            if word in wordvec:
                t = 1
                if word in idf:
                    t = idf[word]
                vec_lst.append((wordvec[word], tf * t))
                sum_tfidf += tf * t
            else:
                unk_cnt += 1
                #sum_tfidf += 3
        sum_tfidf += min(unk_cnt, 1) * 3
        #vec_lst.append((wordvec['UNK'], unk_cnt))
        #print('unk', unk_cnt)
        x = np.zeros(shape=[embedding_size], dtype=np.float)
        if sum_tfidf == 0:
            print(l)
        for vec, weight in vec_lst:
            x += np.array(vec) * (weight / sum_tfidf)
        X.append(x.tolist())
        Y.append(flag)
    return X, Y


def do_classify(wordvec, embedding_size, train_pos_lst, train_neg_lst, test_pos_lst, test_neg_lst):
    idf = get_idf(train_pos_lst + train_neg_lst)
    train_pos_x, train_pos_y = build_data(wordvec, embedding_size, train_pos_lst, 0, idf)
    train_neg_x, train_neg_y = build_data(wordvec, embedding_size, train_neg_lst, 1, idf)
    test_pos_x, test_pos_y = build_data(wordvec, embedding_size, test_pos_lst, 0, idf)
    test_neg_x, test_neg_y = build_data(wordvec, embedding_size, test_neg_lst, 1, idf)
    train_x = train_pos_x + train_neg_x
    train_y = train_pos_y + train_neg_y
    test_x = test_pos_x + test_neg_x
    test_y = test_pos_y + test_neg_y
    print(test_y)
    #predict_y = bayes_classification(train_x, train_y, test_x)
    #predict_y = svm_classification(train_x, train_y, test_x)
    #predict_y = ada_classification(train_x, train_y, test_x)
    #predict_y = rf_classification(train_x, train_y, test_x)
    predict_y = kn_classification(train_x, train_y, test_x)
    print(predict_y.tolist())
    print(precision(test_y, predict_y))


def precision(test_y, predict_y):
    right_num = 0
    for i in range(len(test_y)):
        if test_y[i] == predict_y[i]:
            right_num += 1
    return right_num / len(predict_y)


if __name__ == '__main__':
    pass
    #do_classify()
