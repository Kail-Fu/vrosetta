import networkx as NX
import pygraphviz as PG
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
import numpy
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import OrderedDict 

import nltk
nltk.download('stopwords')
np.set_printoptions(threshold=sys.maxsize)
stopwords = stopwords.words('english')


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
    unique_english_text = list(OrderedDict.fromkeys(english_text))

    all_text_list = [f + ' ' + v + ' ' + e + ' ' for f,
                     v, e in zip(french_text, viet_text, unique_english_text)]

    with open('File.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for data in unique_english_text:
            # print(data)
            writer.writerow([data])

    ####################################################
    # model = SentenceTransformer('average_word_embeddings_komninos')
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # model = SentenceTransformer(
    #     'sentence-transformers/multi-qa-mpnet-base-dot-v1')
    sentence_embeddings = model.encode(unique_english_text)
    # cos_all = cosine_similarity(sentence_embeddings)
    num_clusters = 10
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(sentence_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(unique_english_text[sentence_id])
    # for i, cluster in enumerate(clustered_sentences):
    #     print("Cluster ", i+1)
    #     print(cluster)
    #     print("")

    # Converting to length that fits display
    # english_text_converted_to_fit_display = []
    # maximum_length_per_line = 80
    # for e in english_text:
    #     if len(e) > maximum_length_per_line:
    #         to_add_number = len(e) // maximum_length_per_line
    #         new_text = ""
    #         for _ in range(to_add_number):
    #             new_text += e[:maximum_length_per_line] + "\n"
    #             e = e[maximum_length_per_line:]
    #         english_text_converted_to_fit_display.append(
    #             new_text+e[:maximum_length_per_line])
    #     else:
    #         english_text_converted_to_fit_display.append(e)

    # # H Clustering for all labels
    cos_all = cosine_similarity(sentence_embeddings)
    dif_all = numpy.zeros((len(cos_all), len(cos_all[0])))
    for i in range(len(cos_all)):
    	for j in range(len(cos_all[0])):
    		if i == j:
    			dif_all[i][j] = 1
    		else:
    			dif_all[i][j] = 1-cos_all[i][j]

    X = csr_matrix(dif_all)
    Tcsr = minimum_spanning_tree(X)
    edge_row_indices, edge_col_indices = Tcsr.nonzero()
    nonzero_value_list = Tcsr[edge_row_indices, edge_col_indices]


    G = PG.AGraph()

    for i in range(len(cos_all)):
    	G.add_node(unique_english_text[i], size="small")

    for i in range(len(edge_row_indices)):
    	start = unique_english_text[edge_row_indices[i]]
    	end = unique_english_text[edge_col_indices[i]]
    	distance = nonzero_value_list[0,i]

    	G.add_edge(start, end, len=distance*20)

    G.draw('mst_by_caption.png', format='png', prog='neato')
    
    # for i in range(len(cos_all)):
    # 	G.add_node(i)

    # for i in range(len(edge_row_indices)):
    # 	start = edge_row_indices[i]
    # 	end = edge_col_indices[i]
    # 	distance = nonzero_value_list[0,i]

    # 	G.add_edge(start, end, len=distance*20)

    # G.draw('mst_by_index.png', format='png', prog='neato')


if __name__ == "__main__":
    main()

