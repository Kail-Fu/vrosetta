from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer
import csv

# Replace the path to your own
FINAL_COMPLIED_SHEET = '/Users/fkl/vr/dataset/Final_Compiled Captions, Chars and Links Included_Updated 20221108.xlsx'

def main():
    df = pd.ExcelFile(FINAL_COMPLIED_SHEET).parse("Sheet1")
    english_text = df['English Translation'].tolist()
    english_text = [t if type(t) == str else "" for t in english_text]
    french_text = df['French Text']
    french_text = [t if type(t) == str else "" for t in french_text]
    viet_text = df['Char Text']
    viet_text = [t if type(t) == str else "" for t in viet_text]
    model_for_english_french = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    model_for_viet = SentenceTransformer('keepitreal/vietnamese-sbert')
    english_sentence_embeddings = model_for_english_french.encode(english_text)
    french_sentence_embeddings = model_for_english_french.encode(french_text)
    viet_sentence_embeddings = model_for_viet.encode(viet_text)

    english_cos_all = cosine_similarity(english_sentence_embeddings)
    with open('english/english_similarity_matrix_pm_mpnet_notext.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        count = 0
        for matrix in english_cos_all:
            matrix = list(matrix)
            writer.writerows([matrix])
            count += 1

    with open('english/english_similarity_matrix_pm_mpnet.csv', 'w', newline='') as csvfile:
        tags_space = ['Tags']
        dict_writer = csv.DictWriter(csvfile, tags_space + viet_text)
        dict_writer.writeheader()
        writer = csv.writer(csvfile)
        count = 0
        for matrix in english_cos_all:
            matrix = list([viet_text[count]]+list(matrix))
            writer.writerows([matrix])
            count += 1

    french_cos_all = cosine_similarity(french_sentence_embeddings)
    with open('french/french_similarity_matrix_pm_mpnet_notext.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        count = 0
        for matrix in french_cos_all:
            matrix = list(matrix)
            writer.writerows([matrix])
            count += 1

    with open('french/french_similarity_matrix_pm_mpnet.csv', 'w', newline='') as csvfile:
        tags_space = ['Tags']
        dict_writer = csv.DictWriter(csvfile, tags_space + viet_text)
        dict_writer.writeheader()

        writer = csv.writer(csvfile)
        count = 0
        for matrix in french_cos_all:
            matrix = list([viet_text[count]]+list(matrix))
            writer.writerows([matrix])
            count += 1

    viet_cos_all = cosine_similarity(viet_sentence_embeddings)
    with open('viet/viet_similarity_matrix_viet_sbert_notext.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        count = 0
        for matrix in viet_cos_all:
            matrix = list(matrix)
            writer.writerows([matrix])
            count += 1

    with open('viet/viet_similarity_matrix_viet_sbert.csv', 'w', newline='') as csvfile:
        tags_space = ['Tags']
        dict_writer = csv.DictWriter(csvfile, tags_space + viet_text)
        dict_writer.writeheader()

        writer = csv.writer(csvfile)
        count = 0
        for matrix in viet_cos_all:
            matrix = list([viet_text[count]]+list(matrix))
            writer.writerows([matrix])
            count += 1


if __name__ == "__main__":
    main()
