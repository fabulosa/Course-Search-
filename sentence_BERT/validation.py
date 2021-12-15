import numpy as np
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


if __name__=='__main__':
    course2vec = np.load('course_embeddings_sentBert.npy')
    with open('../course_id.pkl', 'rb') as f:
        course_dict = pickle.load(f)
    course_id = course_dict['course_id']
    id_course = course_dict['id_course']
    similarity = cosine_similarity(course2vec)
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




