import umap
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv


reducer = umap.UMAP()

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
            colors.append(color)

            simi_row = row[14:]
            X.append(simi_row)
        ct += 1
scaled_penguin_data = StandardScaler().fit_transform(X)
embedding = reducer.fit_transform(scaled_penguin_data)

plt.figure(figsize=(100, 100), dpi=200)
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors,
    s=50)
plt.gca().set_aspect('equal', 'datalim')
for i, txt in enumerate(caption):
    plt.annotate(txt, (embedding[:, 0][i], embedding[:, 1][i]), fontsize=3)
plt.title("Red: Female\nBlue: Male\nGreen: Mixed/Nuetral/NA",
          fontdict={'fontsize': 50})
plt.savefig('umap.pdf', format="pdf")
