import codecs

corpus_dir = ""
text_dir = ""
text_dir2 = ""
stopwords_dir = ""

test_files = ["yuhaoming.txt"]
pos_files = ["pos_sentiment.txt", "pos_comments.txt"]
neg_files = ["neg_sentiment.txt", "neg_comments.txt"]
judge_files = ["judgement_neg.txt", "judgement_double_neg.txt"]
degree_files = ["degree_level.txt"]


def read_corpus_file(filename):
    lines = []
    with codecs.open(corpus_dir+filename, "r", "gbk") as f:
        for line in f.readlines():
            line = line.strip()
            if 0 < len(line) < 10:
                lines.append(line)
    return lines


def read_hotel_cmts(filetype, num):
    # read hotel comment test dataset
    # return the num-th text in /textdir/filetype

    text = u''
    with codecs.open(text_dir+filetype+"/"+filetype+"."+str(num)+".txt", "r", "gbk") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                text += u" "+line.strip()
    return text.strip()


def read_test_text():
    # read test text in /sentimentAnalysis/sentiment_analysis_dict/test/
    text_list = []
    label = []
    for filename in test_files:
        with codecs.open(text_dir2+filename, "r", "gbk") as f:
            for line in f.readlines():
                if line[0] == '+':
                    label.append(1)
                elif line[0] == '0':
                    label.append(0)
                else:
                    label.append(-1)
                text_list.append(line[2:].strip())
    return text_list, label


def read_stopwords():
    stopwords = []
    with codecs.open(stopwords_dir, "r", "gbk") as f:
        for line in f.readlines():
            line = line.strip()
            stopwords.append(line)
    return stopwords


def load_corpus_data():
    pos_words = []
    neg_words = []
    judge_words = []
    double_judge_words = []
    degree_words = []
    stopwords = []

    for f in pos_files:
        pos_words.extend(read_corpus_file(f))

    for f in neg_files:
        neg_words.extend(read_corpus_file(f))

    for f in degree_files:
        degree_words.extend(read_corpus_file(f))

    judge_words.extend(read_corpus_file(judge_files[0]))
    double_judge_words.extend(read_corpus_file(judge_files[1]))

    stopwords_origin = read_stopwords()
    for word in stopwords_origin:
        # filter words in dictionary
        if word in pos_words or word in neg_words or word in judge_words or word in double_judge_words \
                or word in degree_words:
            pass
        else:
            stopwords.append(word)  # 1358

    return pos_words, neg_words, judge_words, double_judge_words, degree_words, stopwords


if __name__ == '__main__':
    load_corpus_data()
