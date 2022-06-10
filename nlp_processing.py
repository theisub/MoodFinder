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


data_df = get_records()

print(data_df.head())
sentences = [re.sub('[^a-zA-Z0-9 \n\.]',' ',x) for x in data_df['album_mood'].values]
sentences.insert(0,"I've been hiding from this ever since my breakup with girlfriend number two...the last girlfriend I had and the only one of the two who I really had any romanticism with. We kissed, we caressed, we did all the things that kids our age should be doing. We never had sex, and I didn't want sex. I didn't want anything to do with that; I just wanted her. We had a barren, long-distance relationship that climaxed in two nights that I call the best nights of my life, and then crumbled through the electronic message on a familiar teenage friends and dating site. I've never been the same since then, and it's only been for the worst. As I became worse and worse, I was oblivious to my surroundings. I believed in imaginary beings. I believed that I'd never be okay again. I let the small things get in the way of my life. It went on that way for months. I don't think it would've ever ended if I hadn't shaken myself out of it. I still don't know how I mustered up the strength all on my own, but I did. I did it and I did it all by myself.")
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

to_visualize = pd.DataFrame(data={"album_name":data_df["album_name"].values +" - "+ data_df["artist_name"]})

# pca = PCA(n_components=2, random_state=0)
pca = TSNE(n_components=2, learning_rate='auto',init='random',random_state=0)

components = pca.fit_transform(sentence_embeddings[1:])

fig = px.scatter(components,x=0,y=1, color=to_visualize['album_name'])
fig.show()


print(to_visualize.head())
data_df["similarity"] = similarity_array
data_df =data_df.sort_values(by="similarity",ascending=False)
data_df[["album_name","similarity"]].to_csv("similarity_table.csv")
