#!/usr/bin/env python
# coding: utf-8

# In[10]:


from __future__ import with_statement

import logging
import time
import warnings
import sys
import string

import numpy as np
import pandas as pd

from scipy.stats import multinomial,dirichlet
from scipy.special import digamma

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from gensim import utils, corpora
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.models.hdpmodel import HdpTopicFormatter

meanchangethresh = 0.0001
warnings.filterwarnings("ignore")


# In[3]:


def lda_e_step(ids, cts, alpha, expElogbetad, max_iter=1000):

    gamma = np.ones(len(alpha))
    expElogtheta = np.exp(dirichlet_expectation(gamma))
    
    phinorm = np.dot(expElogtheta, expElogbetad) + 1e-100
    counts = np.array(cts)
    for _ in range(max_iter):
        lastgamma = gamma

        gamma = alpha + expElogtheta * np.dot(cts / phinorm, expElogbetad.T) 
        expElogtheta = np.exp(dirichlet_expectation(gamma))
        
        phinorm = np.dot(expElogtheta,  expElogbetad) + 1e-100
        meanchange = mean_absolute_difference(gamma, lastgamma)
        if meanchange < meanchangethresh:
            break


    return  gamma/np.sum(gamma)


# In[4]:


class Global_Prior():
    def __init__(self, m_alpha, m_K, m_gamma, m_D, m_W, chunksize):
        self.m_alpha = m_alpha
        self.m_K = m_K
        self.m_gamma= m_gamma
        self.m_D= m_D
        self.m_W = m_W
        self.chunksize = chunksize
        
        self.scale = self.m_alpha
       
        
        self.G_0 = np.random.gamma(100., 1./100., size=self.m_K+1) 
        self.G_0 = self.G_0/np.sum(self.G_0)
        
        self.update = 0
    
    def update_G_0(self, rhot, mat_phi):
        
        
        
        self.mat_phi = np.copy(mat_phi)
        
        self.new_G_0 = np.ones(self.m_K+1)/(self.m_K+1) 

        self.vector = np.sum(self.mat_phi, axis=1) * self.m_gamma * self.new_G_0         * self.m_D/self.chunksize 
       
        self.vector[0] += self.m_alpha
    
        self.new_G_0 = self.vector/np.sum(self.vector)
   
        self.G_0 = (1-rhot)* self.G_0 + rhot * self.new_G_0
        self.update += 1 
        
  
    def add_new(self, add_no):
        self.m_K = int(self.m_K + add_no)
        iter = 0
        while iter < add_no:

            new =  1 / (1+ self.m_alpha)
            last = self.G_0[0] * new
            self.G_0[0] = self.G_0[0] * (1-new)
        
            self.G_0 = np.append(self.G_0, last)
            iter += 1


# In[12]:


class HdpModel_CATVI():
    def __init__(self, corpus, corpus_test, id2word, max_K = 400, 
                chunksize=256, kappa=1.0, tau=64.0, K=150, alpha=1, m_beta=1, m_gamma=1):
    
        self.corpus = corpus
        self.id2word = id2word
        self.chunksize = chunksize
        self.corpus_test = corpus_test

        self.max_K = max_K
        
        self.m_alpha = alpha
        self.m_beta = m_beta
        self.m_gamma = m_gamma
        
        self.m_tau = tau + 1
        self.m_kappa = kappa

        self.m_W = len(self.id2word)
        self.m_D = len(self.corpus)
            
        self.m_K = K
        
        self.time = 0
        
        
        self.m_updatect = 0  
        self.rhot = pow(self.m_tau + 0, -self.m_kappa)
        
        self.result = pd.DataFrame(columns = ['iter_no', 'K', "time", "likelihood", "perplexity"])

            
    def get_initial(self, m_lambda_init=0):
        
        self.G_0 = Global_Prior(self.m_alpha, self.m_K, self.m_gamma, self.m_D, self.m_W, self.chunksize)
                
        self.effe_list = np.arange(self.m_K+1)
   
        self.m_lambda =  np.zeros(( self.max_K+1, self.m_W))
    
        self.m_dir_exp_lambda = np.exp(dirichlet_expectation(self.m_lambda + self.m_beta))
        
        
        
    def update(self):
        
        self.start_time = round(time.process_time(),0)
        
        self.read_test_data()
        
        
        for chunk in utils.grouper(self.corpus, self.chunksize):
            if self.finish_status():
            
                try:
                    self.read_text(chunk)

                except:
                    continue

                self.update_chunk(chunk) 
            else:
                
                print('Finish')
                break
    
    def finish_status(self):
        
        
        
        return self.time < 18000
        
                 
          
                
                 
    def get_likelihood(self):
        likelihood_total = 0
        word_counts = 0
        
        beta_matrix = normalize(self.m_lambda[self.effe_list] + self.m_beta, norm='l1', axis=1)
        expElogbeta = self.m_dir_exp_lambda[self.effe_list]
         
        
        
        for i in range(self.test_doc_no):
            train_lists = self.test_doc_word_ids_list[i][0]
            train_cts = self.test_doc_word_counts_list[i][0]
            
            test_lists = self.test_doc_word_ids_list[i][1]
            test_cts = self.test_doc_word_counts_list[i][1]
            
            
            gamma = lda_e_step(train_lists, train_cts, self.G_0.G_0 * self.m_gamma, expElogbeta[:, train_lists])
           
            likelihood  = np.sum(test_cts * np.log(np.dot(gamma,beta_matrix[:, test_lists])+1e-100))
          
            
            likelihood_total += likelihood 
            word_counts += sum(test_cts) 
            

        return likelihood_total/word_counts

        
    def update_chunk(self, chunk):
        start_time = round(time.process_time(),0)
        
        self.mat_z = {}
        self.mat_z_avrg = {}
        self.mat_z_sum = {}
        
        self.mat_phi = np.zeros((self.max_K+1, self.chunk_doc_no))
        
        
        for i in range(self.chunk_doc_no): 
            self.update_doc(i)

            
        #delete_list = np.sum(self.mat_phi[self.effe_list], axis=1) <=  0
        #delete_list[0] =False
        #self.delete_empty_list(delete_list)   

        self.rhot =  pow(self.m_tau + self.m_updatect, -self.m_kappa)

        self.update_lambda(self.rhot)
        self.G_0.update_G_0(self.rhot, self.mat_phi[self.effe_list])

        self.m_updatect += 1
        
        end_time = round(time.process_time(),0)
        used_time = end_time-start_time
        self.time += used_time
        
        
        lilelihood = round(self.get_likelihood(),6)
        result = [self.m_updatect,self.m_K, self.time, lilelihood, round(np.exp(-lilelihood),2)]
        
        self.result.loc[self.m_updatect] = result
        
        print(self.result.tail(1))
        
  
                                              
    def update_doc(self, i, max_iter=500):        
        
        self.mat_z[i] = np.zeros(((self.max_K+1), self.chunk_doc_word_no[i]))
        self.mat_z_avrg[i] = np.copy(self.mat_z[i])
        self.mat_z_sum[i] = np.zeros((self.max_K+1))
                    
        ids = self.chunk_doc_word_ids_list[i]
        cts = self.chunk_doc_word_counts_list[i]
        words_no = self.chunk_doc_word_no[i]
        expElogbetad = self.m_dir_exp_lambda[np.ix_(self.effe_list, ids)]
       
        self.vi(i, ids, cts, words_no, expElogbetad, no_iter=1000)
        self.gibbs_samplings(i, ids, cts, words_no, expElogbetad, max_iter=10)
        
        
        iter = 2
        aver_sum = np.copy(self.mat_z_sum[i])
        aver_phi = digamma(self.G_0.G_0 * self.m_gamma + self.mat_z_sum[i][self.effe_list])
        
        while iter < max_iter:
            last_aver_sum = np.copy(aver_sum)
            
            self.gibbs_samplings(i, ids, cts, words_no, expElogbetad, max_iter=1)
                       
            self.mat_z_avrg[i] -= 1/iter * (self.mat_z_avrg[i] - self.mat_z[i])
            
            aver_sum -= 1/iter * (last_aver_sum - self.mat_z_sum[i])
            
            aver_phi -= 1/iter *(aver_phi - digamma(self.G_0.G_0 * self.m_gamma + self.mat_z_sum[i][self.effe_list]))
                       
            iter += 1
            
            meanchange = mean_absolute_difference(aver_sum[self.effe_list], last_aver_sum[self.effe_list])/np.sum(cts)
            if meanchange <  meanchangethresh:
                break
                
        self.mat_phi[self.effe_list, i] = aver_phi - digamma(self.G_0.G_0 * self.m_gamma)
        
        
        if np.sum(self.mat_z_avrg[i][0]) > 0: 
            add_vector = self.mat_z_sum[i][0]
            add_no = 1
            add_list = ids

            self.m_K += add_no
            new_k = find_gap_in_np_array(self.effe_list, add_no)

            self.effe_list = np.sort(self.effe_list.tolist() + new_k)

            self.mat_z_avrg[i][new_k] =  self.mat_z_avrg[i][0]
            self.mat_z_avrg[i][0] = np.zeros_like(self.mat_z_avrg[i][0])
            
            self.mat_z[i][new_k] =  self.mat_z[i][0]
            self.mat_z[i][0] = np.zeros_like(self.mat_z[i][0])
            
            self.mat_phi[new_k, i] = self.mat_phi[0, i]
            self.mat_phi[0, i] = np.zeros_like(self.mat_phi[0, i])
            

            self.G_0.add_new(add_no)
                   
            self.m_lambda[ np.ix_(new_k, add_list)]             +=  self.rhot * self.m_D/self.chunksize * np.array(cts) *  self.mat_z_avrg[i][new_k]
            self.m_dir_exp_lambda[new_k] = np.exp(dirichlet_expectation(self.m_lambda[new_k] + self.m_beta))
        

        
    def gibbs_samplings(self, i, ids, cts, words_no, expElogbetad, max_iter=1):
        iter = 0

        while iter < max_iter:             
            a = np.tile((self.m_gamma * self.G_0.G_0 + self.mat_z_sum[i][self.effe_list])[:, np.newaxis], (1, words_no))
            a -= np.tile(cts, (self.m_K+1, 1)) * self.mat_z[i][self.effe_list]
            

            mat = a * expElogbetad
            pro_mat = normalize(mat, 'l1', axis=0)
            
            mat_z = my_multinomial(pro_mat)

            self.mat_z[i][self.effe_list] = mat_z
            self.mat_z_sum[i][self.effe_list] = np.dot(mat_z, cts)
                                 
            iter += 1

            
    def delete_empty_list(self, delete_list):
        
        delete_no = np.sum(delete_list)
        
        delete_list2 = self.effe_list[delete_list]
        
        if delete_no != 0:  
            self.m_K -= delete_no
            self.effe_list = self.effe_list[np.logical_not(delete_list)]
                
            self.m_lambda[delete_list2] = np.zeros_like(self.m_lambda[delete_list2] ) * self.m_lambda[0,0]
            self.m_dir_exp_lambda[delete_list2] = np.exp(dirichlet_expectation(self.m_lambda[delete_list2] + self.m_beta))
            
            self.mat_phi[delete_list2] = np.zeros((delete_no, self.chunk_doc_no))

            self.G_0.G_0 = self.G_0.G_0[np.logical_not(delete_list)] 
            self.G_0.m_K = int(len(self.G_0.G_0)-1)
 
            self.G_0.G_0[0] = 1-np.sum(self.G_0.G_0[1:])
           
            for i in range(self.chunk_doc_no): 
            
                self.mat_z[i][delete_list2] = np.zeros_like(self.mat_z[i][delete_list2])
                self.mat_z_avrg[i][delete_list2] = np.zeros_like(self.mat_z_avrg[i][delete_list2])
                

    def update_lambda(self, rhot):
        
        self.m_lambda[self.effe_list] -=  rhot*(self.m_lambda[self.effe_list]) 
        
        for i in range(self.chunk_doc_no):
            ids = self.chunk_doc_word_ids_list[i]
            cts = self.chunk_doc_word_counts_list[i]
            self.m_lambda[np.ix_(self.effe_list, ids)] += rhot* (self.m_D /self.chunksize)            * (np.tile(cts, (self.m_K+1, 1)) *  self.mat_z_avrg[i][self.effe_list]) 
        
        self.m_dir_exp_lambda[self.effe_list] = np.exp(dirichlet_expectation(self.m_lambda[self.effe_list]+ self.m_beta))

          
    def vi(self, i, ids, cts, words_no, expElogbetad, no_iter=1000):  
        
        
        alpha = self.G_0.G_0 * self.m_gamma
    
        gamma = np.ones(len(alpha))
        expElogtheta = np.exp(dirichlet_expectation(gamma))
        
        phinorm = np.dot(expElogtheta, expElogbetad) + 1e-100
        counts = np.array(cts)
        for _ in range(no_iter):
            lastgamma = gamma

            gamma = alpha + expElogtheta * np.dot(cts / phinorm, expElogbetad.T) 
            expElogtheta = np.exp(dirichlet_expectation(gamma))

            phinorm = np.dot(expElogtheta,  expElogbetad) + 1e-100
            meanchange = mean_absolute_difference(gamma, lastgamma)
            if meanchange < meanchangethresh:
                break

        pro_mat = np.outer(expElogtheta.T, 1/phinorm) * expElogbetad
        
        
        mat_z = my_multinomial(pro_mat)
                       
        self.mat_z[i][self.effe_list] =  mat_z
        self.mat_z_sum[i][self.effe_list] = np.dot(mat_z, cts)
        

    def read_test_data(self):
        self.read_text(self.corpus_test, test_indicator =True)
        
        self.test_doc_no = self.chunk_doc_no 
        
        self.test_doc_word_ids_list = self.chunk_doc_word_ids_list.copy()
        self.test_doc_word_counts_list = self.chunk_doc_word_counts_list.copy()
     
        self.test_doc_word_no = self.chunk_doc_word_no.copy()
     
            
    
    def read_text(self, chunk, test_indicator = False):
 
        self.chunk_doc_no = 0
        
        self.chunk_doc_word_ids_list = {}
        self.chunk_doc_word_counts_list = {}
 
        self.chunk_doc_word_no = {}
        iter = 0
        for doc in chunk:
                        
            doc_word_ids, doc_word_counts = zip(*doc)
            doc_word_no = len(doc_word_ids)
            
            if test_indicator == False:
                np_doc_word_ids = np.array(doc_word_ids)
                np_doc_word_counts = np.array(doc_word_counts)
                
                selete_list = np_doc_word_counts >= 1
                pro_doc_word_ids = np_doc_word_ids[selete_list]
                pro_doc_word_counts = np_doc_word_counts[selete_list]
                
                
                
                self.chunk_doc_word_ids_list[iter] = pro_doc_word_ids
                self.chunk_doc_word_counts_list[iter] = pro_doc_word_counts
                self.chunk_doc_word_no[iter] = pro_doc_word_ids.shape[0]

                self.chunk_doc_no += 1
                iter += 1 
                
            else: 
                doc_multi_list = []
                for i in range(doc_word_no):
                    j = 0
                    if doc_word_counts[i] >=1:
                        while j < doc_word_counts[i]:
                            doc_multi_list.append(doc_word_ids[i])
                            j += 1

                
                if len(doc_multi_list) >0:
                    doc_multi_list *= 1
                
                test_tain, test_test = train_test_split(doc_multi_list, test_size=0.10)
                
                test_test_dic = {}
                test_train_dic = {}
                for word in test_tain:
                    try:
                        test_train_dic[word] += 1
                    except:
                        test_train_dic[word] = 1
                for word in test_test:
                    try:
                        test_test_dic[word] += 1
                    except:
                        test_test_dic[word] = 1
                
                test_ids = []
                test_counts = []
                train_ids = []
                train_counts = []
                
                for key, value in test_train_dic.items():
                    train_ids.append(key)
                    train_counts.append(value)
                    
                for key, value in test_test_dic.items():
                    test_ids.append(key)
                    test_counts.append(value)
                
                self.chunk_doc_word_ids_list[iter] = (np.array(train_ids), np.array(test_ids))
                self.chunk_doc_word_counts_list[iter] = (np.array(train_counts), np.array(test_counts))
                self.chunk_doc_word_no[iter] = (len(train_ids), len(test_ids))

                self.chunk_doc_no += 1
                
                iter += 1 


# In[13]:


def find_gap_in_np_array(np_array, add_no):
    list_new = []
    iter=1
    while (len(list_new) < add_no):
        if iter not in np_array:
            list_new.append(iter)
        
        iter += 1

    return list_new


# In[14]:


def my_multinomial(probabilities):
   
    for i in range(probabilities.shape[1]):
        probabilities[:,i] = np.random.multinomial(n=1, pvals=probabilities[:,i])
    return probabilities


# In[21]:





# In[ ]:


if __name__ == '__main__':
    directory = sys.argv[1]
    data_path = directory + '/data'
    save_path = directory + '/result_catvi'
    id2word = corpora.Dictionary.load_from_text(data_path+ "/_wordids.txt.bz2")
    mm = corpora.MmCorpus(data_path+ "/_bow.mm")
    print(mm)
    
    index_train = np.load(data_path + '/index_train.npy')
    index_test = np.load(data_path + '/index_test.npy')
    
    model = HdpModel_CATVI(corpus=mm[index_train], corpus_test=mm[index_test],id2word=id2word, max_K =400,                            chunksize=256, kappa=0.6, tau=64, K=100,  alpha=5, m_beta=5, m_gamma=5)
    model.get_initial()
    model.update()
    
    hdp_formatter = HdpTopicFormatter(model.id2word, model.m_lambda[model.effe_list])
    
    np.save(save_path + '/G_0', model.G_0.G_0)
    
    model.result.index.name = 'iter_no'
    model.result.to_csv(save_path + '/cvi.csv')
    
    dictionary={}
    with open(data_path +'/diffs.txt') as f:
        for line in f.readlines():
            pair = line.split()

            try:
                stemmed = pair[1]
                natural = pair[0]
            except:
                continue
            try:
                dictionary[stemmed] = dictionary[stemmed] + [natural]
            except:
                dictionary[stemmed] = [natural]

    Topics = pd.DataFrame(columns=np.arange(30))

    for item in hdp_formatter.show_topics(500,30):
        text = item[1]
        text = utils.to_unicode(text, 'utf8')
        text = text.translate(str.maketrans('','',string.punctuation))
        text = text.translate(str.maketrans('','',string.digits))
        text_natural = ' '
        for word in text.split():
            try: 
                text_natural += (dictionary[word][0]) + ' '
            except:
                text_natural += word + ' '

        Topics.loc[item[0]] = text_natural.split()

    Topics.to_csv(save_path + '/topics.csv') 
    
