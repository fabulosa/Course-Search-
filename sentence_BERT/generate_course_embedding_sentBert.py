import pandas as pd
import numpy as np
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
np.random.seed(42)
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import pickle
import json
import torch
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
#device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device='cpu'
print("Running on {}".format(device))


def process_batch_data(batch_course, courseId_description_dict):

    batch_description = []

    for i in batch_course:
        print(i)
        batch_description.append(courseId_description_dict[str(i)] if str(i) in courseId_description_dict else '')
    return batch_description


def gene_course_embedding():
    training_data = list(range(len(course_id)))
    train_loader = Data.DataLoader(dataset=training_data, batch_size=128, shuffle=False, num_workers=0,
                                   drop_last=False, collate_fn=lambda x: x)
    course_embeddings = []
    for _, batch_course in enumerate(train_loader):
        batch_description = process_batch_data(batch_course, courseId_description_dict)

        course_embedding = model.encode(batch_description)
        course_embedding = course_embedding.tolist()

        course_embeddings.extend(course_embedding)
    return course_embeddings


if __name__=='__main__':
    with open('../course_id.pkl', 'rb') as f:
        course_dict = pickle.load(f)
        course_id = course_dict['course_id']
        id_course = course_dict['id_course']

    with open('../courseId_description.json', 'r') as f:
        courseId_description_dict = json.load(f)

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    final_course_embeddings = gene_course_embedding()
    np.save("course_embeddings_sentBert.npy", final_course_embeddings)

