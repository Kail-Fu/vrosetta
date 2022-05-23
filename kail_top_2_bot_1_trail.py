from xmlrpc.client import SafeTransport
from bs4 import BeautifulSoup
import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import json
from scipy.spatial import distance
from scipy.spatial.distance import euclidean as eul
from scipy.spatial.distance import cdist as cdist
import re
from langdetect import detect
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AffinityPropagation
import distance
from nltk.stem import WordNetLemmatizer
import nltk
import stringdist
from scipy.spatial.distance import squareform
import random
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import math
from math import floor
from decimal import Decimal
from sklearn.cluster import KMeans
from matplotlib.pyplot import figure

import nltk
nltk.download('stopwords')
np.set_printoptions(threshold=sys.maxsize)
stopwords = stopwords.words('english')

# sentences = [
#     'This is a foo bar sentence.',
#     'This sentence is similar to a foo bar sentence.',
#     'This is another string, but it is not quite similar to the previous ones.',
#     'I am also just another string.'
# ]


def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])

    return text


def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)

    return cosine_similarity(vec1, vec2)[0][0]


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def main():
    #######################################
    df = pd.read_csv(r'../henri.csv')
    # print(df)
    df1 = df.replace(np.nan, ' ', regex=True)

    df_select_columns = df1[['Page Number', ' Image Number (char_id or table_id)',
                             'Language of Caption', 'Caption', 'Viet Translation', 'English Translation']]

    page_1_text = df_select_columns.loc[(df_select_columns['Page Number'] == 1) & (
        df_select_columns['Language of Caption'] == 'fr_text')]
    fr_text_1 = page_1_text['Caption'].to_list()
    vi_text_1 = page_1_text['Viet Translation'].to_list()
    en_text_1 = page_1_text['English Translation'].to_list()
    all_p1 = [f + ' ' + v + ' ' + e + ' ' for f, v,
              e in zip(fr_text_1, vi_text_1, en_text_1)]
    # print(all_p1)

    page_2_text = df_select_columns.loc[(df_select_columns['Page Number'] == 2) & (
        df_select_columns['Language of Caption'] == 'fr_text')]
    fr_text_2 = page_2_text['Caption'].to_list()
    vi_text_2 = page_2_text['Viet Translation'].to_list()
    en_text_2 = page_2_text['English Translation'].to_list()
    all_p2 = [f + ' ' + v + ' ' + e + ' ' for f, v,
              e in zip(fr_text_2, vi_text_2, en_text_2)]
    # print(all_p2)

    alls = []
    alls.append(all_p1)
    alls.append(all_p2)

    all_text = df1.loc[df['Language of Caption'] == 'fr_text']

    french_text = all_text['Caption'].to_list()
    viet_text = all_text['Viet Translation'].to_list()
    english_text = all_text['English Translation'].to_list()
    # print(english_text)
    image_indices = all_text[' Image Number (char_id or table_id)'].to_list()

    all_text_list = [f + ' ' + v + ' ' + e + ' ' for f,
                     v, e in zip(french_text, viet_text, english_text)]

    with open('File.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for data in english_text:
            # print(data)
            writer.writerow([data])

    ####################################################
    # model = SentenceTransformer('average_word_embeddings_komninos')
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # model = SentenceTransformer(
    #     'sentence-transformers/multi-qa-mpnet-base-dot-v1')
    sentence_embeddings = model.encode(english_text)
    # cos_all = cosine_similarity(sentence_embeddings)
    num_clusters = 10
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(sentence_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(english_text[sentence_id])

    cos_all = cosine_similarity(sentence_embeddings)

    linked = linkage(cos_all, 'ward')

    # Directed Force Graph
    derived_clusters = fcluster(linked, t=50, criterion="distance")
    the_obj = {}

    identifiable_english_text = []
    counter_for_clusters = 1
    for e in english_text:
        identifiable_english_text.append(e+"_"+str(counter_for_clusters))
        counter_for_clusters += 1

    nodes_list = []
    for i, c in enumerate(derived_clusters):
        node = {}
        node["id"] = identifiable_english_text[i]
        node["group"] = int(c)
        nodes_list.append(node)
    the_obj["nodes"] = nodes_list

    links_list = []
    df = pd.read_csv("kail_similarity.csv")
    for i, t in enumerate(english_text):
        df_top = df.nlargest(3, [t])
        for row in df_top.index[1:]:  # first 2, except itself
            link = {}
            link["source"] = identifiable_english_text[i]
            link["target"] = identifiable_english_text[row]
            link["value"] = df.iat[i, row]
            links_list.append(link)
    for i, t in enumerate(english_text):
        df_top = df.nsmallest(1, [t])
        for row in df_top.index:  # last 1
            link = {}
            link["source"] = identifiable_english_text[i]
            link["target"] = identifiable_english_text[row]
            link["value"] = df.iat[i, row]
            links_list.append(link)
    the_obj["links"] = links_list

    with open('top_2_bot_1.json', 'w') as outfile:
        json.dump(the_obj, outfile)


if __name__ == "__main__":
    main()
