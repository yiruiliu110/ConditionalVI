"""
Reference: https://github.com/linkstrife/HDP Lihui Lin. School of Data and Computer Science, Sun Yat-sen University
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import codecs
import utils
import scipy.special
from collections import Counter

def load_data(data_url):
    data = []
    with codecs.open(data_url) as data_input:
        for line in data_input:
            doc = []
            for token in line.strip().split():
                doc.append(int(token)-1)
            data.append(doc)
    data_input.close()

    return data

def normalize_row(matrix):
    m_matrix = np.zeros((len(matrix), len(matrix[0])), dtype=np.float32)
    for i, row in enumerate(matrix):
        if np.sum(row) == 0:
            m_matrix[i] = row
            continue
        else:
            total_count = np.sum(row)
            for j, freq in enumerate(row):
                m_matrix[i][j] = freq / total_count

    return m_matrix

def normalize_single_row(row):
    if np.sum(row) == 0:
        row = row
    else:
        total_count = np.sum(row)
        for i, freq in enumerate(row):
            row[i] = freq / total_count

    return row

def corpus_level_G(K, gamma, eta):
    eta = np.full((K,K), eta) # eta defines a uniform distribution, hence symmetric Dirichlet
    m_gamma = np.full(K, gamma)
    mix_components = tfp.distributions.Dirichlet(eta, validate_args=True).sample() # K samples from the base measure H (sym. Dir(eta))
    beta = tfp.distributions.Beta(np.ones(K), m_gamma).sample() # top layer betas (beta')
    with tf.Session():
        m_components = mix_components.eval()
        m_beta = beta.eval()
        stick_remain = 1.0

        for k in range(K):
            if k == 0:
                m_beta[k] = m_beta[k]
            else:
                stick_remain *= (1 - m_beta[k - 1])
                m_beta[k] *= stick_remain

    G = {}
    for k, component in enumerate(m_components):
        G[k] = (m_beta[k], component)

    return G

def document_level_G(G0, alpha):
    betas = []
    components = []
    K = len(G0)
    m_alpha = np.full(K, alpha)

    for k in range(K):
        betas.append(G0[k][0])
        components.append(G0[k][1])

    betas_sum = np.zeros(K)
    for k, beta in enumerate(betas):
        betas_sum[k] = 1 - np.sum(betas[:k])

    pi = tfp.distributions.Beta(m_alpha * betas, m_alpha * betas_sum).sample() # top layer betas (beta')

    with tf.Session():
        m_pi = pi.eval()
        stick_remain = 1.0

        for k in range(K):
            if k==0:
                m_pi[k] = m_pi[k]
            else:
                stick_remain *= (1-m_pi[k-1])
                m_pi[k] *= stick_remain

    G = {}
    for k, component in enumerate(components):
        G[k] = (m_pi[k], component)

    return G

def inference(Iter, GibbsIter, data_set, vocab_size, word_count, gamma, eta, alpha, Gd, burn_in):
    K = len(Gd) # K is very large

    doc_topic_matrix = np.zeros((len(data_set), K), dtype=np.int32)
    topic_word_matrix = np.zeros((K, vocab_size), dtype=np.int32)

    # stochastic initialization
    # for i, doc in enumerate(data_set):
    #    for word in doc:
    #        t = np.random.randint(0, K-1)
    #        k = np.random.randint(0, K-1)
    #        table_topic_map[t] = k
    #        topic_table_count[k] += 1
    #        doc_topic_matrix[i][k] += 1
    #        topic_word_matrix[k][int(word)-1] += 1
    #        table_count_matrix[t][int(word)-1] += 1

    for epoch_iter in range(Iter):
        for i, doc in enumerate(data_set):
            table_count_matrix = np.zeros((K, vocab_size), dtype=np.int32)  # word counts in each table
            topic_table_count = np.zeros(K, dtype=np.int32)  # number of tables belong to topic
            table_topic_map = np.zeros(K, dtype=np.int32)  # table:topic
            for j, token in enumerate(doc):
                    t_list = np.zeros(K, dtype=np.int32) # random initialization with table 1
                    t_list[K-1] = 1
                    t_table = np.zeros((GibbsIter, K))
                    print("Epoch {:d} | Sampling table for token {:s}...".format(epoch_iter+1, token))

                    # —————— Gibbs Sampling for Tables ——————
                    for it in range(GibbsIter + burn_in):
                        # print("Sampling table for {:s} | iter: {:d} | burn-in: {:d}".format(token, it+1, burn_in))
                        for t in range(K): # probabilities for selecting tables
                            t_list[t] = 0 # remove current table assignment
                            unique, counts = np.unique(t_list, return_counts=True)
                            # truncate unassigned table
                            if unique[0] == 0: # there is table with count 0 (current table)
                                unique = np.delete(unique, 0) # delete current table
                                counts = np.delete(counts, 0) # delete current table assignment
                            counts = np.append(counts, alpha) # append new table. [all table assignments, new table]
                            unique = np.append(unique, [max(unique) + 1]) # append new table index
                            u = np.random.uniform() * np.sum(counts)
                            for n, p in enumerate(counts):
                                if np.sum(counts[:n+1]) > u:
                                    t_list[t] = unique[n]
                                    break
                        # new_table[i] = old_table[i]+1, for i in range(len(old_table))
                        old_table = np.unique(t_list) # the table indexes
                        new_table = np.array(range(1, len(old_table) + 1)) # add a new table index
                        for k in range(len(old_table)):
                            # for all elements in t_list that equals to old_table[k], replace it with new_table[k]
                            t_list[t_list == old_table[k]] = new_table[k] # mapping table index, new_index = old_index+1
                        if it >= burn_in:
                            t_table[it - burn_in, :] = t_list
                    sample_list = [Counter(samples).most_common(1)[0][0] for samples in t_table[burn_in:GibbsIter, :]]
                    sampled_t = int(Counter(sample_list).most_common(1)[0][0])
                    # sample_list = np.ceil(np.sum(t_table[burn_in:GibbsIter,:], axis=0) / GibbsIter)
                    # sampled_t = int(Counter(sample_list).most_common(1)[0][0])
                    print("Sampled table {:d}, topic {:d}.".format(int(sampled_t), table_topic_map[int(sampled_t-1)]+1))

                    if sampled_t > K:
                        sampled_t = np.random.randint(0, K)
                        print("Table upper bound reached. Assigned to a random table {:d}.".format(sampled_t))

                    if np.sum(table_count_matrix[sampled_t]) == 0:
                        print("Sampling topic for empty table {:d}...".format(sampled_t))

                        # —————— Gibbs Sampling for Topics ——————
                        # print("Sampling topic for new table {:d} | Gibbs iter: {:d} | burn-in: {:d}".format(sampled_t, it+1, burn_in))
                        sampled_topic = gibbs_sampling_topic(GibbsIter, gamma, topic_word_matrix) - 1
                        print("Sampled topic:", sampled_topic + 1)
                        table_count_matrix[sampled_t, int(token) - 1] += 1
                        doc_topic_matrix[i, table_topic_map[sampled_t]] += 1
                        topic_word_matrix[table_topic_map[sampled_t], int(token) - 1] += 1
                        table_topic_map[sampled_t-1] = sampled_topic  # map new table to a topic
                        # update counting information
                        topic_table_count[sampled_topic] += 1
                        doc_topic_matrix[i, sampled_topic] += 1
                        topic_word_matrix[sampled_topic, int(token) - 1] += 1
                    else:
                        table_count_matrix[sampled_t, int(token) - 1] += 1
                        doc_topic_matrix[i, table_topic_map[sampled_t]] += 1
                        topic_word_matrix[table_topic_map[sampled_t], int(token) - 1] += 1

                    if j < len(doc)-2:
                        print('—————  Next token  —————')

        norm_doc_topic_matrix = normalize_row(doc_topic_matrix)
        norm_topic_word_matrix = normalize_row(topic_word_matrix)

        print('Epoch {:d} | Perplexity: '.format(epoch_iter+1),
              get_perplexity(data_set, norm_doc_topic_matrix, norm_topic_word_matrix, word_count))

        if epoch_iter != Iter-1:
            print('—————  Next epoch  —————')

        if epoch_iter == Iter-1:
            return norm_doc_topic_matrix, norm_topic_word_matrix

def gibbs_sampling_topic(GibbsIter, Gamma, topic_word_matrix):
    K = len(topic_word_matrix)

    z_list = np.zeros(K, dtype=np.int32)  # random initialization with table 1
    z_list[K-1] = 1
    z_table = np.zeros((GibbsIter, K))

    for it in range(GibbsIter + burn_in):
        for k in range(K):
            z_list[k] = 0
            unique, counts = np.unique(z_list, return_counts=True)
            if unique[0] == 0:
                unique = np.delete(unique, 0)
                counts = np.delete(counts, 0)
            counts = np.append(counts, Gamma)  # append new topic.
            unique = np.append(unique, [max(unique) + 1])  # append new topic index
            u = np.random.uniform() * np.sum(counts)
            for j, p in enumerate(counts):
                if np.sum(counts[:j + 1]) > u:
                    z_list[k] = unique[j]
                    break
        old_table = np.unique(z_list)  # the topic indexes
        new_table = np.array(range(1, len(old_table) + 1))  # add a new table index
        for k in range(len(old_table)):
            z_list[z_list == old_table[k]] = new_table[k]  # mapping table index
        if it >= burn_in:
            z_table[it - burn_in, :] = z_list
    sample_list = [Counter(samples).most_common(1)[0][0] for samples in z_table[burn_in:GibbsIter, :]]
    sampled_z = int(Counter(sample_list).most_common(1)[0][0])

    return sampled_z


def get_perplexity(text_data, doc_topic_mat, topic_word_mat, word_count):
    global_topic_dist = np.zeros((1,len(doc_topic_mat[1])))
    global_topic_dist[0] = normalize_single_row(np.sum(doc_topic_mat, axis=0))
    for d, doc_topic in enumerate(doc_topic_mat):
        doc_topic_mat[d] = doc_topic/np.linalg.norm(doc_topic, ord=1)

    with tf.compat.v1.Session():
        p_wd = tf.matmul(tf.convert_to_tensor(doc_topic_mat, dtype=tf.float64), # D x K
                         tf.convert_to_tensor(topic_word_mat, dtype=tf.float64)).eval() # K x V

    sum_log_pw = 0.0
    for d, doc in enumerate(text_data):
        for t, token in enumerate(doc):
            sum_log_pw += np.log(p_wd[d, t])
    perplexity = np.exp(-1 * sum_log_pw / np.sum(word_count))

    return np.around(perplexity, decimals=3)

class hdpModel(object):
    def __init__(self, K, gamma, eta, alpha, Iter, GibbsIter, burn_in, data, vocab_size, vocab_path):
        self.K = K
        self.Iter = Iter
        self.GibbsIter = GibbsIter
        self.burn_in = burn_in
        self.gamma = gamma
        self.alpha = alpha
        self.eta = eta

        self.data, self.word_count = utils.create_seq_data_set_new(data)
        self.word_id_dict = utils.load_word_id(vocab_path)
        self.vocab_size = vocab_size

        self.G0 = corpus_level_G(self.K, self.gamma, self.eta)
        self.Gd = document_level_G(self.G0, self.alpha)

        self.doc_topic_mat, self.topic_word_mat = inference(self.Iter, self.GibbsIter, self.data, self.vocab_size,
                                                            self.word_count, self.gamma, self.eta, self.alpha, self.Gd,
                                                            self.burn_in)

    def get_Corpus_G(self):
        return self.G0

    def get_Document_G(self):
        return self.Gd

    def get_doc_topic_matrix(self):
        return self.doc_topic_mat

    def get_topic_word_matrix(self):
        return self.topic_word_mat

    def print_topics(self, N=-1, num_word=0):
        doc_topic_word = []
        for topic in self.topic_word_mat:
            topic_prob_map = {} # id:prob ascend
            for token, word_prob in enumerate(topic):
                topic_prob_map[token] = word_prob
            topic_prob_map = {token:word_prob for token, word_prob in
                              sorted(topic_prob_map.items(), key=lambda x : x[1])} # id:prob ascend
            doc_topic_word.append(topic_prob_map)
            #print(topic_prob_map)

        topic_list=[]
        for items in doc_topic_word: # id:prob
            topic = []
            for wid in items.keys():
                if np.sum([v for v in items.values()]) != 0.0:
                    topic.append((self.word_id_dict[int(wid)+1], format(float(items[wid]), '.5f'))) # id:prob
            if len(topic) != 0:
                topic_list.append(topic)

        for topics in topic_list[:N]: # N topics
            print((topics[-num_word:])[::-1])

if __name__ == '__main__':
    """
    @param:
    num_topic: upper bound of mixture components
    gamma: top-level concentration parameter for G0 = DP(gamma,H)
    eta: concentration parameter for base measure H (symmetric Dirichlet distribution)
    alpha: second-level concentration parameter for Gd = DP(alpha, G0)
    epoch: number of epoch iterations
    gibbs_iter: number of Gibbs sampling iterations
    burn-in: number of early samples to be truncated
    The last two parameters are data set path and vocabulary size respectively
    """
    num_topic = 10
    gamma = 1.5
    eta = 0.01
    alpha = 1.0
    epoch = 1
    gibbs_iter = 2
    burn_in = 1


    from gensim import corpora
    folder_path = "C:\\Users\\yirui\\OneDrive\ -\ London\ School\ of\ Economics\\CATVI"
    corpus = corpora.MmCorpus(folder_path + "\\arXiV" + "\\data\\_bow.mm")

    from sklearn.model_selection import train_test_split
    index_train, index_test = train_test_split(list(range(len(corpus))), test_size=2000)

    data = corpus[index_test]

    vocab_path = folder_path + "\\arXiV" + "\\data\\ladc.vocab"

    hdp = hdpModel(num_topic, gamma, eta, alpha, epoch, gibbs_iter, burn_in, data, 2000, vocab_path)
    hdp.print_topics(5, 10) # required #topics & #topic words