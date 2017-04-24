import json
import logging
import os
import time

import numpy as np
import tensorflow as tf

import data_utils
from as_reader_tf import AttentionSumReaderTf
from attention_sum_reader import AttentionSumReader

# 基础参数
tf.app.flags.DEFINE_bool(flag_name="debug",
                         default_value=False,
                         docstring="是否在debug模式")

tf.app.flags.DEFINE_bool(flag_name="train",
                         default_value=True,
                         docstring="进行训练")

tf.app.flags.DEFINE_bool(flag_name="test",
                         default_value=False,
                         docstring="进行测试")

tf.app.flags.DEFINE_bool(flag_name="ensemble",
                         default_value=False,
                         docstring="进行集成模型的测试")

tf.app.flags.DEFINE_integer(flag_name="random_seed",
                            default_value=1007,
                            docstring="随机数种子")

tf.app.flags.DEFINE_string(flag_name="log_file",
                           default_value=None,
                           docstring="是否将日志存储在文件中")

tf.app.flags.DEFINE_string(flag_name="weight_path",
                           default_value="model/",
                           docstring="之前训练的模型权重")

tf.app.flags.DEFINE_string(flag_name="framework",
                           default_value="tensorflow",
                           docstring="使用的模型框架，“tensorflow”或者“keras”")

# 定义数据源
tf.app.flags.DEFINE_string(flag_name="data_dir",
                           default_value="D:/source/data/RC-Cloze-CBT/CBTest/CBTest/data/",
                           docstring="CBT数据集的路径")

tf.app.flags.DEFINE_string(flag_name="output_dir",
                           default_value="tmp",
                           docstring="临时目录")

tf.app.flags.DEFINE_string(flag_name="train_file",
                           default_value="cbtest_NE_train.txt",
                           docstring="CBT的训练文件")

tf.app.flags.DEFINE_string(flag_name="valid_file",
                           default_value="cbtest_NE_valid_2000ex.txt",
                           docstring="CBT的验证文件")

tf.app.flags.DEFINE_string(flag_name="test_file",
                           default_value="cbtest_NE_test_2500ex.txt",
                           docstring="CBT的测试文件")

tf.app.flags.DEFINE_string(flag_name="embedding_file",
                           default_value="D:/source/data/embedding/glove.6B/glove.6B.200d.txt",
                           docstring="glove预训练的词向量文件")

tf.app.flags.DEFINE_integer(flag_name="max_vocab_num",
                            default_value=100000,
                            docstring="词库中存储的单词最大个数")

tf.app.flags.DEFINE_integer(flag_name="d_len_min",
                            default_value=0,
                            docstring="载入样本中文档的最小长度")

tf.app.flags.DEFINE_integer(flag_name="d_len_max",
                            default_value=1500,
                            docstring="载入样本中文档的最大长度")

tf.app.flags.DEFINE_integer(flag_name="q_len_min",
                            default_value=0,
                            docstring="载入样本中问题的最小长度")

tf.app.flags.DEFINE_integer(flag_name="q_len_max",
                            default_value=60,
                            docstring="载入样本中问题的最大长度")

# 模型超参数
tf.app.flags.DEFINE_integer(flag_name="hidden_size",
                            default_value=128,
                            docstring="RNN隐层数量")

tf.app.flags.DEFINE_integer(flag_name="num_layers",
                            default_value=1,
                            docstring="RNN层数")

tf.app.flags.DEFINE_bool(flag_name="use_lstm",
                         default_value="False",
                         docstring="RNN类型：LSTM或者GRU")

# 模型训练超参数
tf.app.flags.DEFINE_integer(flag_name="embedding_dim",
                            default_value=200,
                            docstring="词向量维度")

tf.app.flags.DEFINE_integer(flag_name="batch_size",
                            default_value=32,
                            docstring="batch_size")

tf.app.flags.DEFINE_integer(flag_name="num_epoches",
                            default_value=100,
                            docstring="epoch次数")

tf.app.flags.DEFINE_float(flag_name="dropout_rate",
                          default_value=0.2,
                          docstring="dropout比率")

tf.app.flags.DEFINE_string(flag_name="optimizer",
                           default_value="ADAM",
                           docstring="优化算法：SGD或者ADAM或者RMSprop")

tf.app.flags.DEFINE_float(flag_name="learning_rate",
                          default_value=0.001,
                          docstring="SGD的学习率")

tf.app.flags.DEFINE_integer(flag_name="grad_clipping",
                            default_value=10,
                            docstring="梯度截断的阈值，防止RNN梯度爆炸")

FLAGS = tf.app.flags.FLAGS
# bucket，用来处理序列长度方差过大问题
d_bucket = ([150, 310], [310, 400], [450, 600], [600, 750], [750, 950])
q_bucket = (20, 40)


def train_and_test():
    # 准备数据
    vocab_file, idx_train_file, idx_valid_file, idx_test_file = data_utils.prepare_cbt_data(
        FLAGS.data_dir, FLAGS.train_file, FLAGS.valid_file, FLAGS.test_file, FLAGS.max_vocab_num,
        output_dir=FLAGS.output_dir)

    # 读取数据
    d_len_range = (FLAGS.d_len_min, FLAGS.d_len_max)
    q_len_range = (FLAGS.q_len_min, FLAGS.q_len_max)
    t_documents, t_questions, t_answer, t_candidates = data_utils.read_cbt_data(idx_train_file,
                                                                                d_len_range,
                                                                                q_len_range,
                                                                                max_count=FLAGS.max_count)
    v_documents, v_questions, v_answers, v_candidates = data_utils.read_cbt_data(idx_valid_file,
                                                                                 d_len_range,
                                                                                 q_len_range,
                                                                                 max_count=FLAGS.max_count)
    test_documents, test_questions, test_answers, test_candidates = data_utils.read_cbt_data(idx_test_file,
                                                                                             max_count=FLAGS.max_count)
    d_len = data_utils.get_max_length(t_documents)
    q_len = data_utils.get_max_length(t_questions)

    logging.info("-" * 50)
    logging.info("Building model with {} layers of {} units.".format(FLAGS.num_layers, FLAGS.hidden_size))

    # 初始化词向量矩阵，使用(-0.1,0.1)区间内的随机均匀分布
    word_dict = data_utils.load_vocab(vocab_file)
    embedding_matrix = data_utils.gen_embeddings(word_dict,
                                                 FLAGS.embedding_dim,
                                                 FLAGS.embedding_file,
                                                 init=np.random.uniform)

    if FLAGS.framework == "keras":
        # 使用keras版本的模型
        model = AttentionSumReader(word_dict, embedding_matrix, d_len, q_len,
                                   FLAGS.embedding_dim, FLAGS.hidden_size, FLAGS.num_layers,
                                   FLAGS.weight_path, FLAGS.use_lstm)
    else:
        # 使用tensorflow版本的模型
        sess = tf.Session()
        model = AttentionSumReaderTf(word_dict, embedding_matrix, d_len, q_len, sess,
                                     FLAGS.embedding_dim, FLAGS.hidden_size, FLAGS.num_layers,
                                     FLAGS.weight_path, FLAGS.use_lstm)

    if FLAGS.train:
        logging.info("Start training.")
        model.train(train_data=(t_documents, t_questions, t_answer, t_candidates),
                    valid_data=(v_documents, v_questions, v_answers, v_candidates),
                    batch_size=FLAGS.batch_size,
                    epochs=FLAGS.num_epoches_new,
                    opt_name=FLAGS.optimizer,
                    lr=FLAGS.learning_rate,
                    grad_clip=FLAGS.grad_clipping)

    if FLAGS.test:
        logging.info("Start testing.\nTesting in {} samples.".format(len(test_answers)))
        model.load_weight()
        model.test(test_data=(test_documents, test_questions, test_answers, test_candidates),
                   batch_size=FLAGS.batch_size)
    if FLAGS.ensemble:
        logging.info("Start ensemble testing.\nTesting in {} samples.".format(len(test_answers)))
        models = get_ensemble_model(word_dict, embedding_matrix, FLAGS.hidden_size, FLAGS.num_layers, FLAGS.use_lstm)
        ensemble_test((test_documents, test_questions, test_answers, test_candidates), models)


def ensemble_test(test_data, models):
    data = [[] for _ in d_bucket]
    for test_document, test_question, test_answer, test_candidate in zip(*test_data):
        if len(test_document) <= d_bucket[0][0]:
            data[0].append((test_document, test_question, test_answer, test_candidate))
            continue
        if len(test_document) >= d_bucket[-1][-1]:
            data[len(models) - 1].append((test_document, test_question, test_answer, test_candidate))
            continue
        for bucket_id, (d_min, d_max) in enumerate(d_bucket):
            if d_min < len(test_document) < d_max:
                data[bucket_id].append((test_document, test_question, test_answer, test_candidate))
                continue

    acc, num = 0, 0
    for i in range(len(models)):
        num += len(data[i])
        logging.info("Start testing.\nTesting in {} samples.".format(len(data[i])))
        acc_i, _ = models[i].test(zip(*data[i]), batch_size=1)
        acc += acc_i
    logging.critical("Ensemble test done.\nAccuracy is {}".format(acc / num))


def get_ensemble_model(word_dict,
                       embedding_matrix,
                       hidden_size,
                       num_layers,
                       use_lstm):
    embedding_dim = len(embedding_matrix[0])
    models = {}
    for b_id, r in enumerate(d_bucket):
        weight_path = "{}{}-{}-{}-{}/".format(FLAGS.weight_path, r[0], r[1], q_bucket[0], q_bucket[1])
        model = AttentionSumReader(word_dict, embedding_matrix, r[1], q_bucket[1],
                                   embedding_dim, hidden_size, num_layers,
                                   weight_path, use_lstm)
        logging.info(weight_path)
        model.load_weight(weight_path)
        models[b_id] = model
    return models


def clear():
    """
    清除所有临时文件，请谨慎使用
    :return: 
    """
    tmp_dir = os.path.join(FLAGS.data_dir, FLAGS.output_dir)
    if tf.gfile.Exists(tmp_dir):
        tf.gfile.DeleteRecursively(tmp_dir)


def save_arguments(args, file):
    with open(file, "w") as fp:
        json.dump(args, fp, sort_keys=True, indent=4)


def main(_):
    train_and_test()


if __name__ == '__main__':
    # 设置随机数种子
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)
    FLAGS.max_count = 35200 if FLAGS.debug else None
    FLAGS.num_epoches_new = 2 if FLAGS.debug else FLAGS.num_epoches
    # 设置Log
    logging.basicConfig(filename=FLAGS.log_file,
                        filemode='w', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%y-%m-%d %H:%M')
    save_arguments(FLAGS.__flags, "{}args-{}.json".format(FLAGS.weight_path,
                                                          time.strftime("%Y-%m-%d-(%H-%M)",
                                                                        time.localtime())))
    logging.info(FLAGS.__flags)
    tf.app.run()
