__author__ = 'jwj'
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
import pickle
import json


def process_all_descriptions(data):
    # tokenize
    data['title_des'] = data.apply(lambda row: word_tokenize(row['title_des']), axis=1)

    # remove stop words
    stop = stopwords.words('english')
    data["title_des"] = data["title_des"].apply(lambda x: [item.lower() for item in x])
    data['title_des'] = data['title_des'].apply(lambda x: [item for item in x if item not in stop])

    # remove punctuations
    data['title_des'] = data['title_des'].apply(lambda x: [''.join(c for c in s if c not in string.punctuation) for s in x])

    #remove digits
    data['title_des'] = data['title_des'].apply(lambda x: [c for c in x if not c.isdigit()])

    # remove empty string
    data['title_des'] = data['title_des'].apply(lambda x: [s for s in x if s])

    # word lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    data['title_des'] = data['title_des'].apply(lambda x: [wordnet_lemmatizer.lemmatize(w) for w in x])

    # word stemming
    porter_stemmer = PorterStemmer()
    data['title_des'] = data['title_des'].apply(lambda x: [porter_stemmer.stem(w) for w in x])

    data['title_des'] = data['title_des'].apply(' '.join)
    data['title_des'] = data['title_des'].astype(str)

    return data


def generate_word_vector(data):
    vectorizer = CountVectorizer()
    tf = vectorizer.fit_transform(data['title_des'])
    word_dict = {}
    for i in vectorizer.get_feature_names():
        word_dict[i] = len(word_dict)
    with open('word_dict.json', 'w') as f:
        json.dump(word_dict, f)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(tf)
    tfidf = tfidf.toarray()
    return tfidf


if __name__ == '__main__':
    with open('../courseId_description.json', 'r') as f:
        description = json.load(f)
    data = pd.DataFrame(description.items(), columns=['id', 'title_des'])
    data = process_all_descriptions(data)
    tfidf = generate_word_vector(data)
    np.save('tfidf.npy', tfidf)