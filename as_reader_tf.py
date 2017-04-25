import logging
import random
import sys
# noinspection PyUnresolvedReferences
import time

# noinspection PyPep8Naming
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell

_EPSILON = 10e-8


class AttentionSumReaderTf(object):
    def __init__(self,
                 word_dict,
                 embedding_matrix,
                 d_len,
                 q_len,
                 sess,
                 embedding_dim,
                 hidden_size,
                 num_layers,
                 weight_path,
                 use_lstm=False):
        """
        初始化模型
        b ... batch_size
        t ... d_len
        f ... hidden_size*2
        i ... candidate_len 
        """
        self.weight_path = weight_path
        self.word_dict = word_dict
        self.vocab_size = len(embedding_matrix)
        self.d_len = d_len
        self.q_len = q_len
        self.sess = sess
        self.A_len = 10

        logging.info("Embedding matrix shape:%d x %d" % (len(embedding_matrix), embedding_dim))

        self.rnn_cell = LSTMCell(num_units=hidden_size, ) if use_lstm else GRUCell(num_units=hidden_size)
        self.cell_name = "LSTM" if use_lstm else "GRU"

        # 声明词向量矩阵
        with tf.device("/cpu:0"):
            embedding = tf.Variable(initial_value=embedding_matrix,
                                    name="embedding_matrix_w",
                                    dtype="float32")
        # 模型的输入输出
        self.q_input = tf.placeholder(dtype=tf.int32, shape=(None, self.q_len), name="q_input")
        self.d_input = tf.placeholder(dtype=tf.int32, shape=(None, self.d_len), name="d_input")
        self.context_mask_bt = tf.placeholder(dtype=tf.float32, shape=(None, self.d_len), name="context_mask_bt")
        self.candidates_bi = tf.placeholder(dtype=tf.int32, shape=(None, self.A_len), name="candidates_bi")
        self.y_true = tf.placeholder(shape=(None, self.A_len), dtype=tf.float32, name="y_true")

        # 模型输入的长度，每个sample一个长度 shape=(None)
        d_lens = tf.reduce_sum(tf.sign(tf.abs(self.d_input)), 1)
        q_lens = tf.reduce_sum(tf.sign(tf.abs(self.q_input)), 1)

        with tf.variable_scope('q_encoder', initializer=tf.contrib.layers.xavier_initializer()):
            # 问题的编码模型
            # output shape: (None, max_q_length, embedding_dim)
            q_embed = tf.nn.embedding_lookup(embedding, self.q_input)
            q_cell = MultiRNNCell(cells=[self.rnn_cell] * num_layers)
            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw=q_cell,
                                                                   cell_fw=q_cell,
                                                                   dtype="float32",
                                                                   sequence_length=q_lens,
                                                                   inputs=q_embed,
                                                                   swap_memory=True)
            # q_encoder output shape: (None, hidden_size * 2)
            q_encode = tf.concat([last_states[0][-1], last_states[1][-1]], axis=-1)
            logging.info("q_encode shape {}".format(q_encode.get_shape()))

        with tf.variable_scope('d_encoder', initializer=tf.contrib.layers.xavier_initializer()):
            # 上下文文档的编码模型
            # output shape: (None, max_d_length, embedding_dim)
            d_embed = tf.nn.embedding_lookup(embedding, self.d_input)
            d_cell = MultiRNNCell(cells=[self.rnn_cell] * num_layers)
            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw=d_cell,
                                                                   cell_fw=d_cell,
                                                                   dtype="float32",
                                                                   sequence_length=d_lens,
                                                                   inputs=d_embed,
                                                                   swap_memory=True)
            # d_encoder output shape: (None, max_d_length, hidden_size * 2)
            d_encode = tf.concat(outputs, axis=-1)
            logging.info("d_encode shape {}".format(d_encode.get_shape()))

        def att_dot(x):
            """注意力点乘函数"""
            d_btf, q_bf = x
            res = K.batch_dot(tf.expand_dims(q_bf, -1), d_btf, (1, 2))
            return tf.reshape(res, [-1, self.d_len])

        with tf.variable_scope('merge'):
            mem_attention_pre_soft_bt = att_dot([d_encode, q_encode])
            mem_attention_pre_soft_masked_bt = tf.multiply(mem_attention_pre_soft_bt,
                                                           self.context_mask_bt,
                                                           name="attention_mask")
            mem_attention_bt = tf.nn.softmax(logits=mem_attention_pre_soft_masked_bt, name="softmax_attention")

        # 注意力求和，attention-sum过程
        def sum_prob_of_word(word_ix, sentence_ixs, sentence_attention_probs):
            word_ixs_in_sentence = tf.where(tf.equal(sentence_ixs, word_ix))
            return tf.reduce_sum(tf.gather(sentence_attention_probs, word_ixs_in_sentence))

        # noinspection PyUnusedLocal
        def sum_probs_single_sentence(prev, cur):
            candidate_indices_i, sentence_ixs_t, sentence_attention_probs_t = cur
            result = tf.scan(
                fn=lambda previous, x: sum_prob_of_word(x, sentence_ixs_t, sentence_attention_probs_t),
                elems=[candidate_indices_i],
                initializer=tf.constant(0., dtype="float32"))
            return result

        def sum_probs_batch(candidate_indices_bi, sentence_ixs_bt, sentence_attention_probs_bt):
            result = tf.scan(
                fn=sum_probs_single_sentence,
                elems=[candidate_indices_bi, sentence_ixs_bt, sentence_attention_probs_bt],
                initializer=tf.Variable([0] * self.A_len, dtype="float32"))
            return result

        # 注意力求和，output shape: (None, i) i = max_candidate_length = 10
        self.y_hat = sum_probs_batch(self.candidates_bi, self.d_input, mem_attention_bt)

        # 交叉熵损失函数
        output = self.y_hat / tf.reduce_sum(self.y_hat,
                                            reduction_indices=len(self.y_hat.get_shape()) - 1,
                                            keep_dims=True)
        # manual computation of crossentropy
        epsilon = tf.convert_to_tensor(_EPSILON, output.dtype.base_dtype, name="epsilon")
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        self.loss = tf.reduce_mean(- tf.reduce_sum(self.y_true * tf.log(output),
                                                   reduction_indices=len(output.get_shape()) - 1))

        # 计算准确率
        self.correct_prediction = tf.reduce_sum(tf.sign(tf.cast(tf.equal(tf.argmax(self.y_hat, 1),
                                                                         tf.argmax(self.y_true, 1)), "float")))
        # 模型序列化工具
        self.saver = tf.train.Saver()

    # noinspection PyUnusedLocal
    def train(self, train_data, valid_data, batch_size, epochs, opt_name, lr, grad_clip):
        """
        模型训练。
        """
        # 对输入进行预处理
        questions_ok, documents_ok, context_mask, candidates_ok, y_true = self.preprocess_input_sequences(train_data)
        v_questions, v_documents, v_context_mask, v_candidates, v_y_true = self.preprocess_input_sequences(valid_data)

        # 定义模型的优化方法
        if opt_name == "SGD":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif opt_name == "ADAM":
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        else:
            raise NotImplementedError("Other Optimizer Not Implemented.-_-||")

        # 梯度裁剪
        grad_vars = optimizer.compute_gradients(self.loss)
        grad_vars = [
            (tf.clip_by_norm(grad, grad_clip), var)
            if grad is not None else (grad, var)
            for grad, var in grad_vars]
        train_op = optimizer.apply_gradients(grad_vars)
        self.sess.run(tf.global_variables_initializer())

        # 载入之前训练的模型
        self.load_weight()

        # 准备验证集数据
        v_data = {self.q_input: v_questions,
                  self.d_input: v_documents,
                  self.context_mask_bt: v_context_mask,
                  self.candidates_bi: v_candidates,
                  self.y_true: v_y_true}

        # early stopping 参数
        best_val_loss, best_val_acc, patience, lose_times = sys.maxsize, 0, 5, 0
        # 开始训练
        corrects_in_epoch, loss_in_epoch = 0, 0
        batch_num, v_batch_num = len(questions_ok) // batch_size, len(v_questions) // batch_size
        batch_idx, v_batch_idx = np.random.permutation(batch_num), np.arange(v_batch_num)
        logging.info("Train on {} batches, {} samples per batch.".format(batch_num, batch_size))
        logging.info("Validate on {} batches, {} samples per batch.".format(v_batch_num, batch_size))
        for step in range(batch_num * epochs):
            # 一个Epoch结束，输出log并shuffle
            if step % batch_num == 0:
                corrects_in_epoch, loss_in_epoch = 0, 0
                logging.info("--------Epoch : {}".format(step // batch_num + 1))
                np.random.shuffle(batch_idx)

            # 获取下一个batch的数据
            _slice = np.index_exp[
                     batch_idx[step % batch_num] * batch_size:(batch_idx[step % batch_num] + 1) * batch_size]
            data = {self.q_input: questions_ok[_slice],
                    self.d_input: documents_ok[_slice],
                    self.context_mask_bt: context_mask[_slice],
                    self.candidates_bi: candidates_ok[_slice],
                    self.y_true: y_true[_slice]}
            # 训练、更新参数,输出当前Epoch的准确率
            loss_, _, corrects_in_batch = self.sess.run([self.loss, train_op, self.correct_prediction],
                                                        feed_dict=data)
            corrects_in_epoch += corrects_in_batch
            loss_in_epoch += loss_ * batch_size
            nums_in_epoch = (step % batch_num + 1) * batch_size
            logging.info("Trained samples in this epoch : {}".format(nums_in_epoch))
            logging.info("Step : {}/{}.\nLoss : {:.4f}.\nAccuracy : {:.4f}".format(step % batch_num,
                                                                                   batch_num,
                                                                                   loss_in_epoch / nums_in_epoch,
                                                                                   corrects_in_epoch / nums_in_epoch))

            # 每200步保存模型并使用验证集计算准确率，同时判断是否early stop
            if step % 200 == 0 and step != 0:
                # 由于GPU显存不够，仍然按batch计算
                val_num, val_corrects, v_loss = 0, 0, 0
                for i in range(v_batch_num):
                    start = v_batch_idx[i % v_batch_num] * batch_size
                    stop = (v_batch_idx[i % v_batch_num] + 1) * batch_size
                    _v_slice = np.index_exp[start:stop]
                    v_data = {self.q_input: v_questions[_v_slice],
                              self.d_input: v_documents[_v_slice],
                              self.context_mask_bt: v_context_mask[_v_slice],
                              self.candidates_bi: v_candidates[_v_slice],
                              self.y_true: v_y_true[_v_slice]}
                    loss_, v_correct = self.sess.run([self.loss, self.correct_prediction], feed_dict=v_data)
                    val_num = val_num + batch_size
                    val_corrects = val_corrects + v_correct
                    v_loss = v_loss + loss_ * batch_size
                val_acc = val_corrects / val_num
                val_loss = v_loss / val_num
                logging.info("Val acc : {:.4f}".format(val_acc))
                logging.info("Val Loss : {:.4f}".format(val_loss))
                if val_acc > best_val_acc or val_loss < best_val_loss:
                    # 保存更好的模型
                    lose_times = 0
                    path = self.saver.save(self.sess,
                                           'model/machine_reading-val_acc-{:.4f}.model'.format(val_acc),
                                           global_step=step)
                    logging.info("Save model to {}.".format(path))
                else:
                    lose_times += 1
                    logging.info("Lose_time/Patience : {}/{} .".format(lose_times, patience))
                    if lose_times >= patience:
                        logging.info("Oh u, stop training.".format(lose_times, patience))
                        exit(0)

    def test(self, test_data, batch_size):
        # 对输入进行预处理
        questions_ok, documents_ok, context_mask, candidates_ok, y_true = self.preprocess_input_sequences(test_data)
        logging.info("Test on {} samples, {} per batch.".format(len(questions_ok), batch_size))

        # 测试
        batch_num = len(questions_ok) // batch_size
        batch_idx = np.arange(batch_num)
        correct_num, total_num = 0, 0
        for i in range(batch_num):
            start = batch_idx[i % batch_num] * batch_size
            stop = (batch_idx[i % batch_num] + 1) * batch_size
            _slice = np.index_exp[start:stop]
            data = {self.q_input: questions_ok[_slice],
                    self.d_input: documents_ok[_slice],
                    self.context_mask_bt: context_mask[_slice],
                    self.candidates_bi: candidates_ok[_slice],
                    self.y_true: y_true[_slice]}
            correct, = self.sess.run([self.correct_prediction], feed_dict=data)
            correct_num, total_num = correct_num + correct, total_num + batch_size
        test_acc = correct_num / total_num
        logging.info("Test accuracy is : {:.5f}".format(test_acc))

    def load_weight(self):
        ckpt = tf.train.get_checkpoint_state('model/')
        if ckpt is not None:
            logging.info("Load model from {}.".format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            logging.info("No previous models.")

    @staticmethod
    def union_shuffle(data):
        d, q, a, A = data
        c = list(zip(d, q, a, A))
        random.shuffle(c)
        return zip(*c)

    def sort_by_length(self, data):
        # TODO: 数据的ndarray按照length排序，加快训练速度
        pass

    def preprocess_input_sequences(self, data, shuffle=True):
        """
        预处理输入：
        shuffle
        PAD/TRUNC到固定长度的序列
        y_true是长度为self.A_len的向量，index=0为正确答案，one-hot编码
        """
        documents, questions, answer, candidates = self.union_shuffle(data) if shuffle else data
        d_lens = [len(i) for i in documents]

        questions_ok = pad_sequences(questions, maxlen=self.q_len, dtype="int32", padding="post", truncating="post")
        documents_ok = pad_sequences(documents, maxlen=self.d_len, dtype="int32", padding="post", truncating="post")
        context_mask = K.eval(tf.sequence_mask(d_lens, self.d_len, dtype=tf.float32))
        candidates_ok = pad_sequences(candidates, maxlen=self.A_len, dtype="int32", padding="post", truncating="post")
        y_true = np.zeros_like(candidates_ok)
        y_true[:, 0] = 1
        return questions_ok, documents_ok, context_mask, candidates_ok, y_true


def orthogonal_initializer(scale=1.1):
    """
    random orthogonal matrices initializer
    """

    def _initializer(shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)

    return _initializer
