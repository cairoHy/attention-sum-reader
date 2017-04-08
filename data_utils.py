import argparse
import codecs
import logging
import os
import re
from collections import Counter
from functools import reduce

import nltk
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 特殊词汇
# padding,start of sentence,end of sentence,unk,end of question
_PAD = "_PAD"
_BOS = "_BOS"
_EOS = "_EOS"
_UNK = "_UNK"
_EOQ = "_EOQ"
_START_VOCAB = [_PAD, _BOS, _EOS, _UNK, _EOQ]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
EOQ_ID = 4

_BLANK = "XXXXX"


def default_tokenizer(sentence):
    _DIGIT_RE = re.compile(r"\d+")
    sentence = _DIGIT_RE.sub("0", sentence)
    sentence = " ".join(sentence.split("|"))
    return nltk.word_tokenize(sentence.lower())


def gen_vocab(data_file, tokenizer=default_tokenizer, old_counter=None, max_count=None):
    """
    分析语料库，根据词频返回词典。
    """
    logging.info("Creating word_dict from data %s" % data_file)
    word_counter = old_counter if old_counter else Counter()
    counter = 0
    with gfile.FastGFile(data_file) as f:
        for line in f:
            counter += 1
            if max_count and counter > max_count:
                break
            tokens = tokenizer(line.rstrip('\n'))
            word_counter.update(tokens)
            if counter % 100000 == 0:
                logging.info("Process line %d Done." % counter)

    # summary statistics
    total_words = sum(word_counter.values())
    distinct_words = len(list(word_counter))

    logging.info("STATISTICS" + "-" * 20)
    logging.info("Total words: " + str(total_words))
    logging.info("Total distinct words: " + str(distinct_words))

    return word_counter


def save_vocab(word_counter, vocab_file, max_vocab_num=None):
    with gfile.FastGFile(vocab_file, "w") as f:
        for word in _START_VOCAB:
            f.write(word + "\n")
        for word in list(map(lambda x: x[0], word_counter.most_common(max_vocab_num))):
            f.write(word + "\n")


def load_vocab(vocab_file):
    if not gfile.Exists(vocab_file):
        raise ValueError("Vocabulary file %s not found.", vocab_file)
    word_dict = {}
    word_id = 0
    for line in codecs.open(vocab_file, encoding="utf-8"):
        word_dict.update({line.strip(): word_id})
        word_id += 1
    return word_dict


def gen_embeddings(word_dict, embed_dim, in_file=None, init=np.zeros):
    """
    为词表建立一个初始化的词向量矩阵，如果某个词不在词向量文件中，会随机初始化一个向量。
    
    :param word_dict: 词到id的映射。
    :param embed_dim: 词向量的维度。
    :param in_file: 预训练的词向量文件。 
    :param init: 对于预训练文件中找不到的词，如何初始化。
    :return: 词向量矩阵。
    """
    num_words = max(word_dict.values()) + 1
    embedding_matrix = init((num_words, embed_dim))
    logging.info('Embeddings: %d x %d' % (num_words, embed_dim))

    if not in_file:
        return embedding_matrix
    assert get_dim(in_file) == embed_dim
    logging.info('Loading embedding file: %s' % in_file)
    pre_trained = 0
    for line in codecs.open(in_file, encoding="utf-8"):
        sp = line.split()
        if sp[0] in word_dict:
            pre_trained += 1
            embedding_matrix[word_dict[sp[0]]] = np.asarray([float(x) for x in sp[1:]], dtype=np.float32)
    logging.info('Pre-trained: %d (%.2f%%)' %
                 (pre_trained, pre_trained * 100.0 / num_words))
    return embedding_matrix


def get_dim(in_file):
    """
    获取预训练的词向量文件的词向量维度
    """
    line = gfile.FastGFile(in_file, mode='r').readline()
    return len(line.split()) - 1


def get_max_length(d_bt):
    lens = [len(i) for i in d_bt]
    return max(lens)


def sentence_to_token_ids(sentence, word_dict, tokenizer=default_tokenizer):
    """
    把句子中的单词转化为相应的ID。
    比如：
        句子：["I", "have", "a", "dog"]
        word_list：{"I": 1, "have": 2, "a": 4, "dog": 7"}
        返回：[1, 2, 4, 7]

    Args:
      sentence: 句子。
      word_dict: 单词->ID的映射列表。
      tokenizer: 分词器。

    Returns: 整数列表。
    """
    return [word_dict.get(token, UNK_ID) for token in tokenizer(sentence)]


def cbt_data_to_token_ids(data_file, target_file, vocab_file, max_count=None):
    """
    将语料库数据id化并存储。
    
    针对CBT数据集，每22行为一个单元
    前20行：带行数的上下文
    第21行：带行数的问题\t答案\t\t候选答案1|候选答案2|...|候选答案n
    第22行：空白
    
    Args:
      data_file: 源数据文件。
      target_file: 目标文件。
      vocab_file: 词库文件。
      max_count:最多转化的行数。
    """
    if gfile.Exists(target_file):
        return
    logging.info("Tokenizing data in {}".format(data_file))
    word_dict = load_vocab(vocab_file)
    counter = 0

    with gfile.FastGFile(data_file, mode="rb") as data_file:
        with gfile.FastGFile(target_file, mode="w") as tokens_file:
            for line in data_file:
                counter += 1
                if counter % 100000 == 0:
                    logging.info("Tokenizing line %d" % counter)
                if max_count and counter > max_count:
                    break
                if counter % 22 == 21:
                    q, a, _, A = line.split("\t")
                    token_ids_q = sentence_to_token_ids(q, word_dict)[1:]
                    token_ids_A = [word_dict.get(a.lower(), UNK_ID) for a in A.rstrip("\n").split("|")]
                    tokens_file.write(" ".join([str(tok) for tok in token_ids_q]) + "\t"
                                      + str(word_dict.get(a.lower(), UNK_ID)) + "\t"
                                      + "|".join([str(tok) for tok in token_ids_A]) + "\n")
                else:
                    token_ids = sentence_to_token_ids(line, word_dict)
                    token_ids = token_ids[1:] if token_ids else token_ids
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_cbt_data(data_dir, train_file, valid_file, test_file, max_vocab_num, output_dir=""):
    """
    准备CBT语料库，建立词库并将数据id化。
    """
    if not gfile.Exists(os.path.join(data_dir, output_dir)):
        os.mkdir(os.path.join(data_dir, output_dir))
    os_train_file = os.path.join(data_dir, train_file)
    os_valid_file = os.path.join(data_dir, valid_file)
    os_test_file = os.path.join(data_dir, test_file)
    idx_train_file = os.path.join(data_dir, output_dir, train_file + ".%d.idx" % max_vocab_num)
    idx_valid_file = os.path.join(data_dir, output_dir, valid_file + ".%d.idx" % max_vocab_num)
    idx_test_file = os.path.join(data_dir, output_dir, test_file + ".%d.idx" % max_vocab_num)
    vocab_file = os.path.join(data_dir, output_dir, "vocab.%d" % max_vocab_num)

    if not gfile.Exists(vocab_file):
        word_counter = gen_vocab(os_train_file)
        word_counter = gen_vocab(os_valid_file, old_counter=word_counter)
        word_counter = gen_vocab(os_test_file, old_counter=word_counter)
        save_vocab(word_counter, vocab_file, max_vocab_num)

    # 建立id表示的train、valid、test文件
    cbt_data_to_token_ids(os_train_file, idx_train_file, vocab_file)
    cbt_data_to_token_ids(os_valid_file, idx_valid_file, vocab_file)
    cbt_data_to_token_ids(os_test_file, idx_test_file, vocab_file)

    return vocab_file, idx_train_file, idx_valid_file, idx_test_file


def read_cbt_data(file, d_len_range=None, q_len_range=None, max_count=None):
    """
    读取id格式的CBT数据文件。
    :param file: 文件名。
    :param q_len_range: 文档长度范围。
    :param d_len_range: 问题长度范围。
    :param max_count: 读取文件的行数，用于测试。 
    :return: (documents,questions,answers,candidates) 每一个都是numpy数组的形式,shape:(num,?)
    """

    def ok(d_len, q_len):
        a_con = (not d_len_range) or (d_len_range[0] < d_len < d_len_range[1])
        b_con = (not q_len_range) or q_len_range[0] < q_len < q_len_range[1]
        return a_con and b_con

    documents, questions, answers, candidates = [], [], [], []
    with tf.gfile.FastGFile(file, mode="r") as f:
        counter = 0
        d, q, a, A = [], [], [], []
        for line in f:
            counter += 1
            if max_count and counter > max_count:
                break
            if counter % 100000 == 0:
                logging.info("Reading line %d in %s" % (counter, file))
            if counter % 22 == 21:
                tmp = line.strip().split("\t")
                q = tmp[0].split(" ") + [EOS_ID]
                a = [1 if tmp[1] == i else 0 for i in d]
                A = [a for a in tmp[2].split("|")]
                A.remove(tmp[1])
                A.insert(0, tmp[1])  # 将正确答案放在候选答案的第一位
            elif counter % 22 == 0:
                if ok(len(d), len(q)):
                    documents.append(d)
                    questions.append(q)
                    answers.append(a)
                    candidates.append(A)
                d, q, a, A = [], [], [], []
            else:
                d.extend(line.strip().split(" ") + [EOS_ID])  # 每句话结尾加上EOS的ID

    d_lens = [len(i) for i in documents]
    q_lens = [len(i) for i in questions]
    avg_d_len = reduce(lambda x, y: x + y, d_lens) / len(documents)
    logging.info("Document average length: %d." % avg_d_len)
    logging.info("Document midden length: %d." % len(sorted(documents, key=len)[len(documents) // 2]))
    avg_q_len = reduce(lambda x, y: x + y, q_lens) / len(questions)
    logging.info("Question average length: %d." % avg_q_len)
    logging.info("Question midden length: %d." % len(sorted(questions, key=len)[len(questions) // 2]))

    return documents, questions, answers, candidates


def test():
    logging.basicConfig(filename=None,
                        filemode='w', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%y-%m-%d %H:%M')

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        type=str,
                        default="D:/source/data/RC-Cloze-CBT/CBTest/CBTest/data/",
                        help="词库文件路径")
    parser.add_argument('--output_dir',
                        type=str,
                        default="tmp",
                        help="id文件存储路径")
    parser.add_argument('--train_file',
                        type=str,
                        default="cbtest_NE_train.txt",
                        help="训练文件")
    parser.add_argument('--valid_file',
                        type=str,
                        default="cbtest_NE_valid_2000ex.txt",
                        help="验证文件")
    parser.add_argument('--test_file',
                        type=str,
                        default="cbtest_NE_test_2500ex.txt",
                        help="测试文件")
    parser.add_argument('--embed_file',
                        type=str,
                        default="D:/source/data/embedding/glove.6B/glove.6B.100d.txt",
                        help="词向量预训练文件")
    parser.add_argument("--max_vocab_num",
                        type=int,
                        default=100000,
                        help="词库数量")
    parser.add_argument("--d_len_range",
                        type=list,
                        default=(400, 450),
                        help="只载入文档在这个长度范围内的样本")
    parser.add_argument("--q_len_range",
                        type=list,
                        default=(15, 35),
                        help="只载入问题在这个长度范围内的样本")

    args = parser.parse_args()

    vocab_file, idx_train_file, idx_valid_file, idx_test_file = prepare_cbt_data(args.data_dir, args.train_file,
                                                                                 args.valid_file, args.test_file,
                                                                                 args.max_vocab_num,
                                                                                 output_dir=args.output_dir)

    documents, questions, answers, candidates = read_cbt_data(idx_test_file, args.d_len_range, args.q_len_range)
    logging.info(len(documents))


if __name__ == '__main__':
    test()
