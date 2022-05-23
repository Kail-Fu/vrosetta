from bs4 import BeautifulSoup
import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import euclidean as eul
from scipy.spatial.distance import cdist as cdist 
import re
from langdetect import detect
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
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

import nltk
nltk.download('stopwords')

stopwords = stopwords.words('english')

sentences = [
    'This is a foo bar sentence.',
    'This sentence is similar to a foo bar sentence.',
    'This is another string, but it is not quite similar to the previous ones.',
    'I am also just another string.'
]

def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])

    return text

cleaned = list(map(clean_string, sentences))
print(cleaned)

vectorizer = CountVectorizer().fit_transform(cleaned)
vectors = vectorizer.toarray()
print(vectors)

csim = cosine_similarity(vectors)
print(csim)

def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)

    return cosine_similarity(vec1, vec2)[0][0]

cosine_sim_vectors(vectors[0], vectors[1])


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

    df_select_columns = df1[['Page Number', ' Image Number (char_id or table_id)', 'Language of Caption', 'Caption', 'Viet Translation', 'English Translation']]
    # print(df_select_columns)
    # grouped = df_select_columns.LanguageofCaption.groupby([df.index, df_select_columns.PageNumber]).apply(', '.join).reset_index(name='Viet Translation')
    # grouped.columns = ['Page Number','Language of Caption','Viet Translation']

    page_1_text = df_select_columns.loc[(df_select_columns['Page Number'] == 1) & (df_select_columns['Language of Caption'] == 'fr_text')]
    fr_text_1 = page_1_text['Caption'].to_list()
    vi_text_1 = page_1_text['Viet Translation'].to_list()
    en_text_1 = page_1_text['English Translation'].to_list()
    all_p1 = [f + ' ' + v + ' ' + e + ' ' for f, v, e in zip(fr_text_1, vi_text_1, en_text_1)]
    # print(all_p1)

    page_2_text = df_select_columns.loc[(df_select_columns['Page Number'] == 2) & (df_select_columns['Language of Caption'] == 'fr_text')]
    fr_text_2 = page_2_text['Caption'].to_list()
    vi_text_2 = page_2_text['Viet Translation'].to_list()
    en_text_2 = page_2_text['English Translation'].to_list()
    all_p2 = [f + ' ' + v + ' ' + e + ' ' for f, v, e in zip(fr_text_2, vi_text_2, en_text_2)]
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

    all_text_list = [f + ' ' + v + ' ' + e + ' ' for f, v, e in zip(french_text, viet_text, english_text)]

    with open('File.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for data in english_text:
            # print(data)
            writer.writerow([data])

    sub_eng_text = english_text[120:140]
    sub_img_indices = image_indices[120:140]
    X = np.array(list(zip(sub_img_indices, sub_eng_text)))
    
    ####################################################
    # print(sub_eng_text)
    with open('sub1.txt', 'w') as f:
        for item in sub_eng_text:
            f.write("%s\n" % item)
    cleaned = list(map(clean_string, sub_eng_text))
    vectorizer = CountVectorizer().fit_transform(cleaned) #converts strings to numerical vectors
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)
    
    # model = AgglomerativeClustering(affinity='precomputed', n_clusters=15, linkage='complete').fit(csim)
    # print(model.labels_)

    ####################################################
    model = SentenceTransformer('average_word_embeddings_komninos')
    sentence_embeddings = model.encode(english_text)
    cos_all = cosine_similarity(sentence_embeddings)
    # cos = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])

    sub_sentence_embeddings = model.encode(sub_eng_text)
    sub_cos = cosine_similarity(sub_sentence_embeddings)

    # print(cos_all)

    # with open('dissimilarity.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for matrix in cos_all:
    #         writer.writerows([matrix])
    
    with open('sub_similarity.csv', 'w', newline='') as csvfile:
        dict_writer = csv.DictWriter(csvfile, sub_eng_text)
        dict_writer.writeheader()

        with open('sub_similarity.csv','r') as read_file:
            reader = csv.reader(read_file, delimiter=',')
            for col in reader:
                col[0] = sub_eng_text

        writer = csv.writer(csvfile)
        for matrix in sub_cos:
            writer.writerows([matrix])

    with open('similarity.csv', 'w', newline='') as csvfile:
        dict_writer = csv.DictWriter(csvfile, english_text)
        dict_writer.writeheader()

        # with open('similarity.csv','r') as read_file:
        #     reader = csv.reader(read_file, delimiter=',')
        #     for col in reader:
        #         col[0] = english_text

        writer = csv.writer(csvfile)
        for matrix in cos_all:
            writer.writerows([matrix])
    
    sub_spec_clus = SpectralClustering(3).fit_predict(sub_cos)
    spec_clus = SpectralClustering(6).fit_predict(cos_all)

    df = pd.read_csv("sub_similarity.csv")
    
    # print(df)
    print("HIGHEST VALUES")
    # most_similar = df.nlargest(5, ['Paper fan.']))
    # most_similar.to_csv(index=False)

    print("HIGHEST VALUES IN WHOLE DATASET")
    df2 = pd.read_csv("similarity.csv")

    # make a new column with values the columns values:
    df2['Tags'] = df2.columns

    # set index to be new column
    df2 = df2.set_index(df2.Tags)
    df2.to_csv('similarity.csv', index=True)
    print(df2)
    # print(df2.nlargest(1000, ['Paper fan.']))
    # most_similar.to_csv(index=False)



    km = KMeans(n_clusters=3, init='random', max_iter=100, n_init=1, verbose=1)
    km.fit(sub_cos)
    labels = km.predict(sub_cos)
    # print(labels)
    
    # print(SpectralClustering(6).fit_predict(cos_all))
    # print(cos_all)


    # create list of sums
    sums = []
    # create dictionary  of tag: [# cluster it exists in, order it is in the list of strings]
    tag_dict = {}
    order_count = 0
    for tag,cluster_num in zip(sub_eng_text, spec_clus):
        tag_dict[tag] = [cluster_num, order_count]
        order_count += 1

    # print(tag_dict)

    for key in tag_dict:
        distance_sum = 0

        cluster_number = tag_dict[key][0]
        order_number = tag_dict[key][1]

        other_cluster_member_orders = []
        for tag in tag_dict:
            if (tag_dict[tag][0] == cluster_number) & (tag != key):
                other_cluster_member_orders.append(tag_dict[tag][1])
        
        for member_order in other_cluster_member_orders:
            distance_sum += cos_all[order_number][member_order]

        sums.append(distance_sum)
    print(sums)

            




    # for every key in dictionary:
        # create sum of distances for the key of interest = 0
        # get # cluster it exists
        # get # order it is in list of strings (cluster x)
        # using # cluster, get dictionary of other strings in that same cluster
        # for every key in that sub-dictionary:
            # find value in cluster x
            # add to sum of distances for the key of interest
        # add to big list of sums (size 20)

        









    # print(sentence_embeddings)
    # print(type(cos_all[0][0]))
    ####################################################

    distances = []

    # print(X[0])
    # d = get_jaccard_sim(X[1], X[1])
    # print(d)
    for i in range(len(X)):
        # print(X[i])
        for j in range(len(X[1])):
            d = get_jaccard_sim(X[i][j], X[i][j])
            # print(d)
            distances.append(d)
    
    rands = []
    for i in range(20):
        r = random.uniform(0, 1)
        rands.append(r)
    # print(rands)
    
    b = np.random.uniform(low=0.0, high=1.0, size=(20,20))
    mat = (b + b.T)/2
    np.fill_diagonal(mat, 0)
    mat.astype(float)
    # print(b_symm)

    # mat2 = np.array([[1.0, 0.0, 0.0, 0.0, 0.25, 0.1, 0.2, 0.6, 0.5, 0.0, 0.0, 0.2, 0.3, 0.5, 0.1, 0.1, 0.0, 0.2, 0.0 , 0.1], 
    #                 [0.0, 1.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.05, 0.7, 0.2, 0.3, 0.1, 0.7, 0.2, 0.3, 0.3, 0.2, 0.2, 0.1, 0.0], 
    #                 [0.0, 0.0, 1.0, 0.1, 0.0, 0.1, 0.1, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.4, 0.2],
    #                 [0.0, 0.1, 0.1, 1.0, 0.2, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
    #                 [0.25, 0.1, 0.0, 0.2, 1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
    #                 [0.1, 0.0, 0.1, 0.3, 0.1, 1.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.2, 0.1, 0.0, 0.3, 0.4, 0.2, 0.3, 0.0, 0.0],
    #                 [0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.1, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.3, 0.0, 0.5],
    #                 [0.6, 0.05, 0.5, 0.0, 0.05, 0.6, 0.1, 1.0 , 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1, 0.2, 0.1, 0.1, 0.0, 0.0],
    #                 [0.5, 0.7, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.2, 0.1, 0.0, 0.2],
    #                 [0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
    #                 [0.2, 0.1, 0.3, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0],
    #                 [0.3, 0.7, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.1, 0.0, 0.2, 0.1, 0.1, 0.0, 0.0],
    #                 [0.5, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.1, 0.3, 0.1, 0.2],
    #                 [0.5, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.1, 0.3, 0.1, 0.2],
    #                 [0.5, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.1, 0.3, 0.1, 0.2],
    #                 [0.5, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.1, 0.3, 0.1, 0.2],
    #                 [0.5, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.1, 0.3, 0.1, 0.2],
    #                 [0.5, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.1, 0.3, 0.1, 0.2],
    #                 [0.5, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.1, 0.3, 0.1, 0.2],
    #                 ])
    # dists = squareform(mat)
    # linkage_matrix = linkage(dists, "single")
    # dendrogram(linkage_matrix, labels=["Blind medium.", "Beggar's gestures.", "Ear cleaner."])
    # dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=8, labels=["Blind medium.", 
    #                                     "Beggar’s gestures.", 
    #                                     "Ear cleaner.", 
    #                                     "Childbirth outside the home.", 
    #                                     "Role of the rooster in the world of magic.", 
    #                                     "Butcher.", 
    #                                     "Plane.", 
    #                                     "Betel nut knife.", 
    #                                     "Peasants returning from market.", 
    #                                     "Rich woman’s outfit.", 
    #                                     "Door Guardian Deity (folk print).", 
    #                                     "Sugar mixing in kẹo candy making.", 
    #                                     "Seller of copper objects.", 
    #                                     "Woman printing.", 
    #                                     "Scraping a coconut.", 
    #                                     "Blacksmith hammering a pot.", 
    #                                     "Planishing sandals.", 
    #                                     "Pastry cook flattening the dough.", 
    #                                     "Mask worn for eye ailments.", 
    #                                     "Ancestors’ shelf, its cover."])
    doubles = []
    doubles_array = []
    for array in cos_all:
        for f in array:
            string = "%.2f" % round(f, 2)
            double = Decimal(string)
            doubles.append(double)
            # print(doubles)
        doubles_array.append(doubles)
        doubles = []
    doubles_np = np.asarray(doubles_array)
    # print(doubles_np)

    #Using the float list (cos_all) here, to use doubles, use doubles_np:
    np.fill_diagonal(cos_all, 0)
    dists2 = squareform(cos_all)
    linkage_matrix_2 = linkage(dists2, "single")
    # print(linkage_matrix_2)



    new_text = [ele[:5] + "..." for ele in sub_eng_text]
    print(sub_eng_text)

    # dendrogram(linkage_matrix_2, leaf_rotation=90, leaf_font_size=8, labels=new_text)
    # dendrogram(linkage_matrix_2, leaf_rotation=90, leaf_font_size=8, labels=["Blind medium.", 
    #                                     "Beggar’s gestures.", 
    #                                     "Ear cleaner.", 
    #                                     "Childbirth...", 
    #                                     "Role of the rooster..."])
    plt.hist(sums, 20)
    plt.title("Sum of Drawing Tag Distances (Subset 1)")
    plt.xlabel("Sum of Distances Between Similarity Values")
    plt.ylabel("Count")
    
    plt.show()



    # lemmatizer = WordNetLemmatizer()

    # for sentence in X:
    #     lemmatizer.lemmatize(sentence)

    # print(X)
    concat = np.array(list(zip(sub_img_indices, distances)))

    new_X = [float(i) for i in concat[:,1]]
    new_X = np.reshape(new_X, (len(new_X), 1))

    Z = linkage(new_X, method='ward')

    c, coph_dists = cophenet(Z, pdist(new_X))



    # words = sub_eng_text.split(" ") #Replace this line
    # words = np.asarray(words) #So that indexing with a list will work
    # lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in sub_eng_text] for w2 in sub_eng_text])

    # affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
    # affprop.fit(lev_similarity)
    # for cluster_id in np.unique(affprop.labels_):
    #     exemplar = sub_eng_text[affprop.cluster_centers_indices_[cluster_id]]
    #     cluster = np.unique(sub_eng_text[np.nonzero(affprop.labels_==cluster_id)])
    #     cluster_str = ", ".join(cluster)
    #     print(" - *%s:* %s" % (exemplar, cluster_str))


    #######################################

    pics = []
    for i in range(1, 61):
        og = i
        webpage = "view"+str(i)+".html"
        if i<10:
            i = "00"+str(i)
        elif (i<100):
            i = "0"+str(i)
        else:
            i = str(i)


        image_file = "../test_images/OGER_Page_"+str(i)+".jpg"
        image = Image.open(image_file)
        ###Add border
        #  image = ImageOps.expand(image, border=20)
        ###

        with open(os.path.join(sys.path[0], webpage), "r") as f:
            html = f.read()
            soup = BeautifulSoup(html, 'html.parser')
            words = soup.find_all('span', attrs={'class':'fn-note-content'})
            p = soup.find_all('span', attrs={'id':'unique-id-dogphotomeg8'})
            # if pics.find_all('img'):
            #     imagesurl = pics.find_all('img').get('image-file')
            #     print(imagesurl)
            # print(p)
            locs = soup.findAll("div", {"class": "fn-area"})
            locations = []
            for loc in locs:
                locations.append(loc['style'])
            old_size = image.size
            # im = im.resize(size_tuple)
            # print(image.size)
            # image = image.resize((1800, 1350))
            # print(image.size)

            new_size = (old_size[0], int(old_size[1]*(2)))
            new_im = Image.new("RGB", new_size, 'white')
            new_im.paste(image, ((0, 0)))
            draw = ImageDraw.Draw(new_im)
            font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', size=35)
            color = 'rgb(0, 0, 0)' # black color
            #old_im = Image.open('someimage.jpg')
        
            #pos_x = 390
            #pos_y = 85
            #for j in range(0, 11):
            position = []
            pos_to_sort = []
            for j in range(len(words)):
                left = int(locations[j].split()[1][:-3])
                top = int(locations[j].split()[3][:-3])
                height = int(locations[j].split()[7][:-3])
                position.append((left, top))
                pos_to_sort.append((eul((left, top), (0, 0))))
                #zero_mat.append((0, 0))
                #distances = np.append(distances, [(left, top+height)])
                #distances[j] = [[left, top+height]]
            # print(words)
            dist = cdist(position, position, 'euclidean')
            #pos_sort = distance.cdist(position, zero_mat, 'euclidean')

            start = np.argsort(pos_to_sort)[0]
            #print(np.argsort(pos_to_sort))
            words_sorted = []
            pos_sorted = []
            already_sorted = [start]
            for j in range(len(words)):
                #print(start)
                words_sorted.append(words[int(start)])
                pos_sorted.append((int(locations[start].split()[1][:-3]), int(locations[start].split()[3][:-3]), int(locations[start].split()[7][:-3])))
                if(j<len(words)-1):
                    count = 1
                    new_start = np.argsort(dist[start])[count]
                    # l = list()
                    while(len(list(filter(lambda x : x == new_start, already_sorted))) > 0):
                        count+=1
                        new_start = np.argsort(dist[start])[count]
                    start = new_start
                    already_sorted.append(start)
            text = str("")
            #f= open("caption"+str(i)+".txt","w+")
            for j in range(len(words)):
                left = pos_sorted[j][0]
                top = pos_sorted[j][1]
                height = pos_sorted[j][2]
                pos_x=left

                pos_y = top+height
                # print((words_sorted[j].get_text()))
                #f.write((str(og)+"."+str(j+1)+") "+words_sorted[j].get_text()).encode('utf-8')+'\n')
                text = text + " ("+str(og)+"."+str(j+1)+") "+(words_sorted[j].get_text())
                # print(detect(words_sorted[j].get_text()))
                if detect(words_sorted[j].get_text()) == 'vi':
                    viet = words_sorted[j].get_text()
                    viet = '\033[1m' + viet
                # print(viet)
                if detect(words_sorted[j].get_text()) == 'fr':
                    french = words_sorted[j].get_text()
                    french = '\033[92m' + french

                draw.text((pos_x*1.875, pos_y*1.82), str(j+1), font=font, color=color)

            font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', size=35)
            max_width = 1800
            lines = []
            if font.getsize(text)[0]  <= max_width:
                lines.append(text)
            else:
                #split the line by spaces to get words
                words = text.split(' ')
                k = 0
                # append every word to a line while its width is shorter than the image width
                while k < len(words):
                    line = ''

                    while k < len(words) and font.getsize(line + words[k])[0] <= max_width:
                        line = line + words[k]+ " "
                        k += 1
                    if not line:
                        line = words[k]
                        k += 1
                    lines.append(line)
            # print(lines)
            x = 10
            y=old_size[1]
            # print(lines)
            # draw.text((x, y), all_p1, fill='black', font=font)
            out_str = ' '
            for a in alls:


                y = y+40
            # draw.text((10, 1000), lines, font=font, fill=(0, 0, 0))
            #     draw.text((x, y), name, fill=color, font=font)


            #new_im.save('annotated_image'+str(i)+'.png')
            pics.append(new_im)

    images = pics
    widths, heights = zip(*(i.size for i in images))



    total_width = sum(widths)
    max_height = max(heights)
    max_width = max(widths)
    #Adjusts canvas width and height:
    new_im = Image.new('RGB', (max_width*7, max_height*2))

    x_offset = 0
    count = 0
    y_offset = 0
    for j in range(14): # was 60
        count+=1
        ###Add border
        images[j] = ImageOps.expand(images[j], border=20)
    #############
    #   images[j].show()
    new_im.paste(images[j], (x_offset, y_offset))
    #   place = Image.open("../test_images/place_spot.png")
    #   new_im.paste(place, (x_offset, y_offset))
  
    if(count==7): # was 20
        x_offset=0
        y_offset += max_height
        count=0
    else:
        x_offset += images[j].size[0]
  

    #print("here")
    #new_im.show()
    print("SAVED!!!!")
    new_im.save('7by2_wall1_1.png')   

if __name__ == "__main__":
    main()
