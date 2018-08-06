import sys
import gensim
import numpy as np
import pickle

glove_vector = {}

class WordEmbedding:
#    def __init__(self,dataset,filepath,datarelated,datastances):
    def __init__(self,dataset,filepath,datastances):
        print("Reading Word Vector Dictionary")
        with open(filepath, encoding="utf8") as f_glove:
            for line in f_glove:
                tmp = line.split()
                l = len(tmp)
                glove_vector[' '.join(tmp[0:l-50])] = list(map(np.float,tmp[l-50:l]))
        print("Finish reading Word Vector Dictionary")

        #Start to embed words
        with open("data.p","wb") as embedding:
            pickle.dump(dataset,embedding)
            
        print("Start embedding\n")
            
        with open(datastances, "wb") as embedded_data_stances:
#            embedded_data_stances=open(datastances,'wb')
            data_related=[]
            data_stances=[]
            for row in dataset:
                #Convert the tokens to GLoVe vectors
                title=row[0]
                body=row[1]
                titleVec=[]
                bodyVec=[]
                for i in range(len(title)):
                    #print(title[i] in wordVec)
                    if title[i] in glove_vector:
                        word=glove_vector[title[i]]
                        #print(wordVec["the"])
                        titleVec.append(word)
                for i in range(len(body)):
                    if body[i] in glove_vector:
                        word=glove_vector[body[i]]
                        bodyVec.append(word)
                if len(row) == 3:
                    label=row[2]
                    if label =="unrelated":
                        label=[0,0,0,1]
                    elif label == "discuss":
                        label=[1,0,0,0]
                    elif label == "agree":
                        label=[0,1,0,0]
                    elif label == "disagree":
                        label=[0,0,1,0]
                    data_stances.append([titleVec,bodyVec,label])
                else:
                    data_stances.append([titleVec,bodyVec])
            print('agree: '+str(a)+'unrelated: '+str(u)+'discuss: '+str(d)+'disagree: '+str(da))
            pickle.dump(np.array(data_stances),embedded_data_stances)
            print("Transfer word to vector successfully\n")
