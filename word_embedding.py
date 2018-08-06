import sys
import gensim
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib import rnn
from preprocess import Preprocess
from embedding import WordEmbedding



if __name__ == "__main__":
    
    train_dataset = []
    test_dataset = []
    val_dataset = []
    
    
    train = Preprocess("data/train_data.csv")
    validation = Preprocess("data/validation_data.csv")
    test = Preprocess("data/test_data.csv")
    body = Preprocess("data/article_body_texts.csv")


    #Merge stances and bodies in train dataset
    for row in train.tokens:
        content = [row[0:len(row)-2]] + [body.tokens[row[len(row)-2]]] + [row[len(row)-1]]
        train_dataset.append(content)

    #Merge stances and bodies in validation dataset
    for row in validation.tokens:
        content = [row[0:len(row)-2]] + [body.tokens[row[len(row)-2]]] + [row[len(row)-1]]
        val_dataset.append(content)

    #Merge stances and bodies in test dataset
    for row in test.tokens:
        content = [row[0:len(row)-1]] + [body.tokens[row[len(row)-1]]]
        test_dataset.append(content)

    train_glove = WordEmbedding(train_dataset,'glove.6B/glove.6B.50d.txt','train_stances.p')
    val_glove = WordEmbedding(val_dataset,'glove.6B/glove.6B.50d.txt','val_stances.p')
    test_glove = WordEmbedding(test_dataset,'glove.6B/glove.6B.50d.txt','test_stances.p')




#    #using Word2vec
#    w2v_model = gensim.models.Word2Vec(
#                                       a.tokens,
#                                       size=300, # Dimension of the word embedding
#                                       window=2, # The maximum distance between the current and predicted word within a sentence.
#                                       min_count=1, # Ignores all words with total frequency lower than this.
#                                       sg=1, # If 1, skip-gram is employed; otherwise, CBOW is used.
#                                       negative=10, # Number of negative samples to be drawn
#                                       iter=20, # Number of epochs over the corpus
#                                       )
#    w2v_model.save('data/w2v_300d_snli_data.pkl')


