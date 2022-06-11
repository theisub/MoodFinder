from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_records
import re
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def find_simillar(sentence):
    data_df = get_records()

    print(data_df.head())
    # sentence = "I am walking through dark forest with thick vegetation,I messed up and end up in deep gooey swamp."
    sentences = [re.sub('[^a-zA-Z0-9 \n\.]',' ',x) for x in data_df['album_mood'].values]
    sentences.insert(0,sentence)
    # sentences =  np.insert(sentences,0,"I am walking through dark forest with thick vegetation,I messed up and end up in deep gooey swamp."

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    sentence_embeddings= sentence_embeddings.numpy()
    print(sentence_embeddings.shape)
    print(data_df["album_name"][0])
    n = 0
    print(cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])[0])
    similarity_array = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])[0]

    # to_visualize = pd.DataFrame(data={"album_name":data_df["album_name"].values +" - "+ data_df["artist_name"]})

    # # pca = PCA(n_components=2, random_state=0)
    # pca = TSNE(n_components=2, learning_rate='auto',init='random',random_state=0)

    # components = pca.fit_transform(sentence_embeddings[1:])

    # fig = px.scatter(components,x=0,y=1, color=to_visualize['album_name'])
    # fig.show()


    # print(to_visualize.head())
    data_df["similarity"] = similarity_array
    return data_df.iloc[np.argmax(similarity_array)]
    # data_df =data_df.sort_values(by="similarity",ascending=False)
    # data_df[["album_name","similarity"]].to_csv("similarity_table.csv")
