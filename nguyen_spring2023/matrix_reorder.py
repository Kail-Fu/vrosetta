import numpy
from scipy.spatial.distance import squareform
from fastcluster import linkage
import csv
from numpy import genfromtxt
import sys
sys.setrecursionlimit(30000)

def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N, 0])
        right = int(Z[cur_index-N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat, finalized_idxs, idx_list, method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''

    N = len(dist_mat)
    flat_dist_mat = squareform(numpy.array(dist_mat))

    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    # Provides an array of column order
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = numpy.zeros((N, N))

    a, b = numpy.triu_indices(N, k=1)

    non_duped = []
    [non_duped.append(x) for x in a if x not in non_duped]
    for i in non_duped:
        finalized_idxs.append(idx_list[res_order[i]])
    [finalized_idxs.append(x) for x in idx_list if x not in finalized_idxs]

    seriated_dist[a, b] = dist_mat[[res_order[i]
                                    for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


# English
english_sorted_list = []
with open('english/english_similarity_matrix_pm_mpnet.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    ct = 0
    for row in reader:
        if ct > 0:
            english_sorted_list.append(row)
        ct += 1
english_number_data = genfromtxt(
    'english/english_similarity_matrix_pm_mpnet_notext.csv', delimiter=',')
english_finalized_idxs = []
english_idx_list = []
for i in range(len(english_sorted_list)):
    english_idx_list.append(i)
N = len(english_number_data)
for i in range(N):
    for j in range(N):
        english_number_data[i][j] = 1 - abs(round((english_number_data[i][j]), 3))
english_dist_mat = english_number_data
english_ordered_dist_mat, english_res_order, english_res_linkage = compute_serial_matrix(english_dist_mat, english_finalized_idxs, english_idx_list, "average")
csv_list = []
for i in range(len(english_ordered_dist_mat)):
    tags = []
    try:
        tags.append(english_finalized_idxs[i])
    except:
        print(i)
        print(len(english_ordered_dist_mat))
    csv_list.append(tags + list(english_ordered_dist_mat[i]))
with open('english/english_reordered_matrix_off_caption_vectors_pm_mpnet.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    count = 0
    for matrix in csv_list:
        matrix = list(matrix)
        writer.writerows([matrix])
        count += 1


# French
french_sorted_list = []
with open('french/french_similarity_matrix_pm_mpnet.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    ct = 0
    for row in reader:
        if ct > 0:
            french_sorted_list.append(row)
        ct += 1
french_number_data = genfromtxt(
    'french/french_similarity_matrix_pm_mpnet_notext.csv', delimiter=',')
french_finalized_idxs = []
french_idx_list = []
for i in range(len(french_sorted_list)):
    french_idx_list.append(i)
N = len(french_number_data)
for i in range(N):
    for j in range(N):
        french_number_data[i][j] = 1 - abs(round((french_number_data[i][j]), 3))
french_dist_mat = french_number_data
french_ordered_dist_mat, french_res_order, french_res_linkage = compute_serial_matrix(french_dist_mat, french_finalized_idxs, french_idx_list, "average")
csv_list = []
for i in range(len(french_ordered_dist_mat)):
    tags = []
    try:
        tags.append(french_finalized_idxs[i])
    except:
        print(i)
        print(len(french_ordered_dist_mat))
    csv_list.append(tags + list(french_ordered_dist_mat[i]))
with open('french/french_reordered_matrix_off_caption_vectors_pm_mpnet.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    count = 0
    for matrix in csv_list:
        matrix = list(matrix)
        writer.writerows([matrix])
        count += 1


# Viet
viet_sorted_list = []
with open('viet/viet_similarity_matrix_viet_sbert.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    ct = 0
    for row in reader:
        if ct > 0:
            viet_sorted_list.append(row)
        ct += 1
viet_number_data = genfromtxt(
    'viet/viet_similarity_matrix_viet_sbert_notext.csv', delimiter=',')
viet_finalized_idxs = []
viet_idx_list = []
for i in range(len(viet_sorted_list)):
    viet_idx_list.append(i)
N = len(viet_number_data)
for i in range(N):
    for j in range(N):
        viet_number_data[i][j] = 1 - abs(round((viet_number_data[i][j]), 3))
viet_dist_mat = viet_number_data
viet_ordered_dist_mat, viet_res_order, viet_res_linkage = compute_serial_matrix(viet_dist_mat, viet_finalized_idxs, viet_idx_list, "average")
csv_list = []
for i in range(len(viet_ordered_dist_mat)):
    tags = []
    try:
        tags.append(viet_finalized_idxs[i])
    except:
        print(i)
        print(len(viet_ordered_dist_mat))
    csv_list.append(tags + list(viet_ordered_dist_mat[i]))
with open('viet/viet_reordered_matrix_off_caption_vectors_viet_sbert.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    count = 0
    for matrix in csv_list:
        matrix = list(matrix)
        writer.writerows([matrix])
        count += 1



