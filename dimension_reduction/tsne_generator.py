"""
==========================
tSNE to visualize digits
==========================

Here we use :class:`sklearn.manifold.TSNE` to visualize the digits
datasets. Indeed, the digits are vectors in a 8*8 = 64 dimensional space.
We want to project them in 2D for visualization. tSNE is often a good
solution, as it groups and separates data points based on their local
relationship.

"""
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import csv
import pandas as pd


############################################################
# Load the iris data
# from sklearn import datasets
# digits = datasets.load_digits()
# X = digits.data[:500]
# y = digits.target[:500]

###
caption = []
X = []
colors = []
with open('../english/english_mpnet_embedding_matrix.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    ct = 0
    for row in reader:
        if (ct > 0):
            caption.append(row[0])

            color = 'green'
            femality = str(row[11])
            if femality == "Female":
                color = 'red'
            if femality == "Male":
                color = 'blue'
            if femality == "Mixed":
                color = "yellow"
            colors.append(color)

            simi_row = row[14:]
            X.append(simi_row)
        ct += 1

############################################################
# Fit and transform with a TSNE
tsne = TSNE(n_components=2, random_state=0)

############################################################
# Project the data in 2D
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(100, 100), dpi=200)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=100)

# fig, ax = plt.subplots(figsize=(100, 100), dpi=200)
# ax.scatter(x, y, c=colors, s=100)
for i, txt in enumerate(caption):
    plt.annotate(txt, (X_2d[:, 0][i], X_2d[:, 1][i]), fontsize=5)

# plt.title("Red: Female\nBlue: Male\nYellow: Mixed\nGreen: Uncertain/NA",
#           fontdict={'fontsize': 50})
plt.savefig('books_read.pdf', format="pdf")


############################################################
# Visualize the data
# target_ids = range(len(digits.target_names))
# target_ids = [0]

# from matplotlib import pyplot as plt
# plt.figure(figsize=(6, 5))
# colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
# # for i, c, label in zip(target_ids, colors, digits.target_names):
# for i, c, label in zip(target_ids, colors, [0]):
#     print(i, c, label)
#     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
# plt.legend()
# plt.show()
