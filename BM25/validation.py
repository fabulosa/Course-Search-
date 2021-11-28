from rank_bm25 import BM25Okapi
import json
import pandas as pd
from IPython.display import HTML
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import pickle
import numpy as np


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


if __name__ == '__main__':
    with open('../courseId_description.json', 'r') as f:
        description = json.load(f)
    data = pd.DataFrame(description.items(), columns=['id', 'title_des'])
    data = process_all_descriptions(data)
    tokenized_corpus = [doc.split(" ") for doc in data['title_des'].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    similarity = []
    i = 0
    for course in tokenized_corpus:
        print(i)
        i+=1
        doc_scores = bm25.get_scores(course)
        similarity.append(doc_scores.tolist())

    similarity = np.array(similarity)

    with open('../course_id.pkl', 'rb') as f:
        course_dict = pickle.load(f)
    course_id = course_dict['course_id']
    id_course = course_dict['id_course']

    vali_set = pd.read_csv('../vali_pairs.csv', header=0)
    vali_set_split1 = vali_set['1'].str.split(' ')
    vali_set_split2 = vali_set['2'].str.split(' ')

    vali_set['course1'] = vali_set_split1.str[1:].apply(lambda x: ' '.join(x)) + ' ' + vali_set_split1.str[0]
    vali_set['course2'] = vali_set_split2.str[1:].apply(lambda x: ' '.join(x)) + ' ' + vali_set_split2.str[0]
    vali_set = vali_set.loc[:, ['course1', 'course2']]
    vali_set['course1'] = vali_set['course1'].apply(lambda x: course_id[x] if x in course_id else -1)
    vali_set['course2'] = vali_set['course2'].apply(lambda x: course_id[x] if x in course_id else -1)
    vali_set = vali_set.loc[(vali_set['course1']!=-1)&(vali_set['course2']!=-1)]

    vali_set = vali_set.to_numpy()

    recall = []
    rank = []
    for i in vali_set:
        sim = similarity[i[0]]
        ranking = list((-sim).argsort())[0:]
        rankk = ranking.index(i[1])
        rank.append(rankk)
        if rankk <= 10:
            recall.append(1)
        else:
            recall.append(0)

    s = pd.Series(rank)
    print(s.describe())
    print("Recall: ", np.average(recall))