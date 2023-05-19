import csv
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

def main():
    similarity_matrix = []
    labels = [] 
    with open('english/english_4454.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        ct = 0

        for row in reader:
            if (ct > 0):
                similarity_matrix.append(row[4:])
                labels.append(row[0])

            ct += 1

    # Compute the linkage matrix
    linkage_matrix = hierarchy.linkage(similarity_matrix, method='average', metric='euclidean')
    plt.figure(figsize=(10, 300))
    plt.rcParams.update({'font.size': 7})


    # Plot the dendrogram
    hierarchy.dendrogram(linkage_matrix, labels=labels, leaf_rotation=0, orientation='right')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering Dendrogram')
    # Rotate and align the labels
    plt.subplots_adjust(left=0.3)

    # Save the dendrogram as PDF
    plt.savefig('hierarchical_clustering.pdf')

if __name__ == "__main__":
    main()