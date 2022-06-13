from gettext import find
from turtle import settiltangle
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_records
import re
import numpy as np
# import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
from django.conf import settings 
import logging



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def find_simillar(sentence):

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.getLogger('find_sim').setLevel(logging.WARNING)

    log = logging.getLogger()

    t1 = time.time()
    data_df = get_records()
    t2 = time.time()
    log.info('records read with ',t2-t1)


    t1 = time.time()
    sentences = [re.sub('[^a-zA-Z0-9 \n\.]',' ',x) for x in data_df['album_mood'].values]
    sentences.insert(0,sentence)
    # sentences =  np.insert(sentences,0,"I am walking through dark forest with thick vegetation,I messed up and end up in deep gooey swamp."
    t2 = time.time()
    log.info('records cleaned and added new one with ',t2-t1)


    t1 = time.time()
    tokenizer = settings.TOKENIZER
    t2 = time.time()
    log.info('read tokenizer from pretrained with ',t2-t1)

    t1 = time.time()
    model = settings.MODEL

    t2 = time.time()
    log.info('read model from pretrained with ',t2-t1)

    t1 = time.time()
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    t2 = time.time()
    log.info('input encoded with ',t2-t1)

    t1 = time.time()
    with torch.no_grad():
        model_output = model(**encoded_input)
    t2 = time.time()
    log.info('gave output with ',t2-t1)


    t1 = time.time()

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    sentence_embeddings= sentence_embeddings.numpy()
    print(sentence_embeddings.shape)
    print(data_df["album_name"][0])
    n = 0
    print(cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])[0])
    similarity_array = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])[0]
    t2 = time.time()
    print('gave result with ',t2-t1)

    data_df["similarity"] = similarity_array
    return data_df.iloc[np.argmax(similarity_array)]

