import logging
import os

import numpy as np
import tensorflow as tf

import data_utils
from attention_sum_reader import AttentionSumReader

# 基础参数
tf.app.flags.DEFINE_bool(flag_name="debug",
                         default_value=False,
                         docstring="是否在debug模式")

tf.app.flags.DEFINE_bool(flag_name="test_only",
                         default_value=False,
                         docstring="只进行测试，不训练")

tf.app.flags.DEFINE_integer(flag_name="random_seed",
                            default_value=1007,
                            docstring="随机数种子")

tf.app.flags.DEFINE_string(flag_name="log_file",
                           default_value=None,
                           docstring="是否将日志存储在文件中")

tf.app.flags.DEFINE_string(flag_name="weight_path",
                           default_value="model/weight",
                           docstring="之前训练的模型权重")

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
                           default_value="D:/source/data/embedding/glove.6B/glove.6B.100d.txt",
                           docstring="glove预训练的词向量文件")

tf.app.flags.DEFINE_integer(flag_name="max_vocab_num",
                            default_value=100000,
                            docstring="词库中存储的单词最大个数")

tf.app.flags.DEFINE_integer(flag_name="d_len_min",
                            default_value=400,
                            docstring="载入样本中文档的最小长度")

tf.app.flags.DEFINE_integer(flag_name="d_len_max",
                            default_value=450,
                            docstring="载入样本中文档的最大长度")

tf.app.flags.DEFINE_integer(flag_name="q_len_min",
                            default_value=15,
                            docstring="载入样本中问题的最小长度")

tf.app.flags.DEFINE_integer(flag_name="q_len_max",
                            default_value=35,
                            docstring="载入样本中问题的最大长度")

# 模型超参数
tf.app.flags.DEFINE_integer(flag_name="hidden_size",
                            default_value=128,
                            docstring="RNN隐层数量")

tf.app.flags.DEFINE_integer(flag_name="num_layers",
                            default_value=3,
                            docstring="RNN层数")

tf.app.flags.DEFINE_bool(flag_name="use_lstm",
                         default_value="False",
                         docstring="RNN类型：LSTM或者GRU")

# 模型训练超参数
tf.app.flags.DEFINE_integer(flag_name="embedding_dim",
                            default_value=100,
                            docstring="词向量维度")

tf.app.flags.DEFINE_integer(flag_name="batch_size",
                            default_value=16,
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


def train():
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
    d_len = data_utils.get_max_length(t_documents)
    q_len = data_utils.get_max_length(t_questions)

    logging.info("-" * 50)
    logging.info("Building model with {} layers of {} units.".format(FLAGS.num_layers, FLAGS.hidden_size))

    # 初始化词向量矩阵
    word_dict = data_utils.load_vocab(vocab_file)
    embedding_matrix = data_utils.gen_embeddings(word_dict, FLAGS.embedding_dim, FLAGS.embedding_file)

    if True:
        # 使用keras版本的模型
        model = AttentionSumReader(word_dict, embedding_matrix, d_len, q_len,
                                   FLAGS.embedding_dim, FLAGS.hidden_size, FLAGS.num_layers,
                                   FLAGS.weight_path, FLAGS.use_lstm)
        logging.info("Start training.")
        model.train(train_data=(t_documents, t_questions, t_answer, t_candidates),
                    valid_data=(v_documents, v_questions, v_answers, v_candidates),
                    batch_size=FLAGS.batch_size,
                    epochs=FLAGS.num_epoches_new,
                    opt_name=FLAGS.optimizer,
                    lr=FLAGS.learning_rate,
                    grad_clip=FLAGS.grad_clipping)
    else:
        # 使用tensorflow版本的模型
        with tf.Session():
            pass


def test():
    pass


def clear():
    """
    清除所有临时文件，请谨慎使用
    :return: 
    """
    tmp_dir = os.path.join(FLAGS.data_dir, FLAGS.output_dir)
    if tf.gfile.Exists(tmp_dir):
        tf.gfile.DeleteRecursively(tmp_dir)


def main(_):
    if not FLAGS.test_only:
        train()
    test()


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
    tf.app.run()
