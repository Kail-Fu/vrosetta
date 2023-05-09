import numpy
from scipy.spatial.distance import squareform
from fastcluster import linkage
import csv
# import matplotlib.pyplot as plt
from numpy import genfromtxt
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import sys
sys.setrecursionlimit(30000)

number_data = []
sorted_list = []
# Reading our CSV
# with open('english/english_similarity_matrix_xlm_r.csv', 'r') as csvfile:
# with open('french/french_similarity_matrix_xlm_r.csv', 'r') as csvfile:
with open('viet/new/viet_similarity_matrix_distiluse2.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    ct = 0

    for row in reader:
        if ct > 0:
            sorted_list.append(row)
        ct += 1

# number_data = genfromtxt(
#     'english/english_similarity_matrix_xlm_r_notext.csv', delimiter=',')
# number_data = genfromtxt(
#     'french/french_similarity_matrix_xlm_r_notext.csv', delimiter=',')
number_data = genfromtxt(
    'viet/new/viet_similarity_matrix_distiluse2_notext.csv', delimiter=',')

finalized_idxs = []
idx_list = []


for i in range(len(sorted_list)):
    idx_list.append(i)

N = len(number_data)
print(N)

for i in range(N):
    for j in range(N):
        number_data[i][j] = 1 - abs(round((number_data[i][j]), 3))

dist_mat = number_data


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


def compute_serial_matrix(dist_mat, method="ward"):
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


methods = ["average"]

for method in methods:
    ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(
        dist_mat, method)

    csv_list = []
    for i in range(len(ordered_dist_mat)):
        tags = []

        try:
            tags.append(finalized_idxs[i])
        except:
            print(i)
            print(len(ordered_dist_mat))
        csv_list.append(tags + list(ordered_dist_mat[i]))

    # with open('english/english_reordered_matrix_off_caption_vectors_xlm_r.csv', 'w', newline='') as csvfile:
        # with open('french/french_reordered_matrix_off_caption_vectors_xlm_r.csv', 'w', newline='') as csvfile:
    with open('viet/new/viet_reordered_matrix_off_caption_vectors_distiluse2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        count = 0
        for matrix in csv_list:
            matrix = list(matrix)
            writer.writerows([matrix])
            count += 1
