import csv
import re
import nltk
import numpy as np
from sklearn import feature_extraction


def normalize_word(w):
    return nltk.stem.WordNetLemmatizer().lemmatize(w)

class Preprocess():
    def __init__(self, fileinfo):
        self.fileinfo = fileinfo
        print("Reading dataset from", fileinfo)
        self.data = self.openfile(fileinfo)
        self.tokens = self.execute(self.data)
   
        
    def execute(self,data):
        # Propocess the data
        articles = {}
        title_tokens = []
        body_tokens = {}
        #make the body ID an integer value
        for s in self.data:
            s['Body ID'] = int(s['Body ID'])

            if 'articleBody' in s:
                clean_data = self.clean(s['articleBody'])
                clean_data = self.get_tokenized_lemmas(clean_data)
                clean_data = self.remove_stopwords(clean_data)
                body_tokens[s['Body ID']] = clean_data

            elif 'Headline' in s:
                clean_data = self.clean(s['Headline'])
                clean_data = self.get_tokenized_lemmas(clean_data)
                clean_data = self.remove_stopwords(clean_data)
                clean_data.append(s['Body ID'])
                if 'Stance' in s:
                    clean_data.append(s['Stance'])
                title_tokens.append(clean_data)
        
        if title_tokens:
            return title_tokens
        elif body_tokens:
            return body_tokens


    def openfile(self, fileinfo):
        # Read files
        with open(fileinfo, encoding='utf-8') as bodies:
            csv_bodies = csv.DictReader(bodies)
            row_content = []
            for row in csv_bodies:
                row_content.append(row)

        return row_content

    def get_tokenized_lemmas(self, s):
        # Tokenize sentences and phrases
        return [normalize_word(t) for t in nltk.word_tokenize(s)]
    

    def clean(self, s):
        # Clean a string: Lowercasing, trimming, removing non-alphanumeric
        return " ".join(re.findall(r'[a-zA-Z]+', s, flags=re.UNICODE)).lower()
    
    def remove_stopwords(self, l):
        # Remove stopwords from a list of tokens
        return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


