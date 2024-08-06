#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tqdm import tqdm

data = pd.read_csv('train.tsv',sep='\t')


# In[3]:


data.describe()


# In[4]:


phrases = data['Phrase'].values
y = data['Sentiment'].values


# In[5]:


def build_vocab(phrases):
    vocab = set()
    for phrase in phrases:
        words = phrase.split() 
        vocab.update(words)
    return list(vocab)

vocab = build_vocab(phrases)


# In[6]:


def text_2_vector(text, vocab):
    vector = np.zeros(len(vocab))
    words = text.split()
    for word in words:
        vector[vocab.index(word)] += 1
    return vector


# In[7]:


X = np.array([text_2_vector(phrase, vocab) for phrase in phrases])


# In[8]:


X_with_bias = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


# In[9]:


def ont_hot(y):
    num_classes = y.max() - y.min() + 1
    return np.eye(num_classes)[y]


# In[10]:


ont_hot_y = ont_hot(y)


# In[11]:


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# In[12]:


def propagation(X,y,W):
    n = X.shape[0]
    z = np.dot(X,W)
    a = softmax(z)
    loss = -np.mean(y*np.log(a))
    dz = a - y
    dw = 1/n * np.dot(X.T, dz)
    return loss, dw


# In[13]:


def train(X, y, epochs=1000, batch_size=10, lr=0.001):
    w = np.random.rand(X.shape[1], 5)
    loss_history = []
    for _ in tqdm(range(epochs)):
        index = np.random.permutation(X.shape[0])
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[index[i:i+batch_size]]
            y_batch = y[index[i:i+batch_size]]
            loss, dw = propagation(X_batch, y_batch, w)
            loss_history.append(loss)
            w = w - dw * lr
    return w, loss


# In[14]:


X_with_bias


# In[ ]:





# In[15]:


train(X_with_bias, ont_hot_y, batch_size=100000)


# In[ ]:





# In[ ]:




