import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_records
import re
import numpy as np
import pandas as pd
from django.conf import settings 


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def find_simillar(sentence):
    data_df = get_records()

    sentences = [re.sub('[^a-zA-Z0-9 \n\.]',' ',x) for x in data_df['album_mood'].values]
    sentences.insert(0,sentence)
    tokenizer = settings.TOKENIZER

    model = settings.MODEL

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)
    

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    sentence_embeddings= sentence_embeddings.numpy()
    similarity_array = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])[0]

    data_df["similarity"] = similarity_array
    return data_df.iloc[np.argmax(similarity_array)]

