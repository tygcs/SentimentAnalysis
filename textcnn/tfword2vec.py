"""
word to vector
"""

import collections
import math
import json
import os

import numpy as np
import tensorflow as tf


class TFWord2Vec(object):
    def __init__(self, data_file, embedding_size, vocab_size, num_skips, skip_window, batch_size, num_sampled, num_steps):
        self.data_file = data_file
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.num_steps = num_steps

    def read_data(self):
        print('read data')
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = f.read()
        self.sentence = data.split('\n')
        return data.split()

    def build_dataset(self):
        words = self.read_data()
        print('build dataset')
        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(words).most_common(self.vocab_size - 1))
        self.vocab = {}
        for word, _ in self.count:
            self.vocab[word] = len(self.vocab)
        self.data = []
        unk_cnt = 0
        for word in words:
            if word in self.vocab:
                idx = self.vocab[word]
            else:
                idx = 0
                unk_cnt += 1
            self.data.append(idx)
        self.count[0][1] = unk_cnt
        self.reversed_vocab = dict(zip(self.vocab.values(), self.vocab.keys()))

    def build_graph(self):
        print('build graph')
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights,
                               biases=self.nce_biases,
                               labels=self.train_labels,
                               inputs=embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocab_size))
            self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / norm
            self.init = tf.global_variables_initializer()

    def train(self):
        average_loss = 0
        for step in range(self.num_steps):
            sen = self.sentence[step % len(self.sentence)]
            batch_inputs = []
            batch_labels = []
            for i in range(len(sen)):
                start = max(0, i - self.num_skips)
                end = min(len(sen), i + self.num_skips + 1)
                for idx in range(start, end):
                    if idx == i:
                        continue
                    else:
                        input_id = self.vocab.get(sen[i])
                        label_id = self.vocab.get(sen[idx])
                        if not (input_id and label_id):
                            continue
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
            if batch_inputs:
                batch_inputs = np.array(batch_inputs, dtype=np.int32)
                batch_labels = np.array(batch_labels, dtype=np.int32)
                batch_labels = np.reshape(batch_labels, [batch_labels.__len__(), 1])
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}
                _, loss_val = self.session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val
                if step % 2000 == 0 and step != 0:
                    average_loss /= 2000
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0
        final_embeddings = self.normalized_embeddings.eval(session=self.session)
        return final_embeddings

    def save_embedding(self, embedding, file):
        print('save embedding')
        d = {}
        for i in range(len(embedding)):
            d[self.reversed_vocab[i]] = embedding[i, :].tolist()
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(d, f)
        return d

    def word2vec(self, wordvec_file, force=False):
        if not force and os.path.exists(wordvec_file):
            with open(wordvec_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        self.build_dataset()
        self.build_graph()
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)
        final_embeddings = self.train()
        return self.save_embedding(final_embeddings, wordvec_file)


if __name__ == '__main__':
    tfwv = TFWord2Vec('sougouca_seg', 100, 50000, 2, 2, None, 200, 100000)
    #wordvec = tfwv.word2vec('wordvec', True)
