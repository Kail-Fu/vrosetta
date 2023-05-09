from scipy.spatial.distance import cdist as cdist
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer
import csv
# import numpy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
# import pygraphviz as PG


def main():
    df = pd.ExcelFile(
        '/Users/fkl/vr/dataset/Final_Compiled Captions, Chars and Links Included_Updated 20221108.xlsx').parse("Sheet1")
    # english_text = df['English Translation'].tolist()
    # english_text = [t if type(t) == str else "" for t in english_text]
    # french_text = df['French Text']
    # french_text = [t if type(t) == str else "" for t in french_text]
    viet_text = df['Char Text']
    viet_text = [t if type(t) == str else "" for t in viet_text]
    # model = SentenceTransformer(
    #     "sentence-transformers/stsb-xlm-r-multilingual")
    # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    # model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    # model = SentenceTransformer('keepitreal/vietnamese-sbert')
    # model = SentenceTransformer(
    #     'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # model = SentenceTransformer(
    #     'sentence-transformers/distiluse-base-multilingual-cased-v2')
    model = SentenceTransformer('keepitreal/vietnamese-sbert')
    # sentence_embeddings = model.encode(english_text)
    # sentence_embeddings = model.encode(french_text)
    sentence_embeddings = model.encode(viet_text)

    cos_all = cosine_similarity(sentence_embeddings)

    with open('viet/new/viet_similarity_matrix_viet_sbert_notext.csv', 'w', newline='') as csvfile:
        # with open('french/french_similarity_matrix_xlm_r_notext.csv', 'w', newline='') as csvfile:
        # with open('english/english_similarity_matrix_xlm_r_notext.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        count = 0
        for matrix in cos_all:
            matrix = list(matrix)
            writer.writerows([matrix])
            count += 1

    with open('viet/new/viet_similarity_matrix_viet_sbert.csv', 'w', newline='') as csvfile:
        # with open('english/english_similarity_matrix_xlm_r.csv', 'w', newline='') as csvfile:
        # with open('french/french_similarity_matrix_xlm_r.csv', 'w', newline='') as csvfile:
        tags_space = ['Tags']
        # dict_writer = csv.DictWriter(csvfile, tags_space + english_text)
        # dict_writer = csv.DictWriter(csvfile, tags_space + french_text)
        dict_writer = csv.DictWriter(csvfile, tags_space + viet_text)
        dict_writer.writeheader()

        writer = csv.writer(csvfile)
        count = 0
        for matrix in cos_all:
            # matrix = list([english_text[count]] + list(matrix))
            # matrix = list([french_text[count]]+list(matrix))
            matrix = list([viet_text[count]]+list(matrix))
            writer.writerows([matrix])
            count += 1


if __name__ == "__main__":
    main()
