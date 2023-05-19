import csv
import numpy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import pygraphviz as PG
from PIL import Image


def main():
    cos_all = []
    unique_text = []  # (caption, color)
    links = []
    empty_list = []
    with open('english/english_4454.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        ct = 0

        for row in reader:
            if (ct > 0):
                if row[0]:
                    cos_all.append(row[4:])
                    links.append(row[3])

                    unique_text.append(
                        [str(ct) + "_" + row[0]+"\n"+row[1]+"\n"+row[2], "black"])  # no color
                else:
                    empty_list.append(ct-1)
            ct += 1

    # remove empty ones
    empty_list.reverse()  # make sure index not in conflict
    [j.pop(a) for a in empty_list for j in cos_all]

    dif_all = numpy.zeros((len(cos_all), len(cos_all[0])))
    for i in range(len(cos_all)):
        for j in range(len(cos_all[0])):
            if i == j:
                dif_all[i][j] = 1
            else:
                dif_all[i][j] = cos_all[i][j]
    print(dif_all.shape)

    X = csr_matrix(dif_all)
    Tcsr = minimum_spanning_tree(X)
    edge_row_indices, edge_col_indices = Tcsr.nonzero()
    nonzero_value_list = Tcsr[edge_row_indices, edge_col_indices]

    G = PG.AGraph()
    # If you want to create graph with less overlapping nodes, use the following one
    # G = PG.AGraph(overlap="prism")
    for i in range(len(cos_all)):
        graph_loc = 'web_low_res/' + links[i].split("/")[-1]
        im = Image.open(graph_loc)
        ideal_height = im.size[1]/67.2+2
        G.add_node(unique_text[i][0], image=graph_loc, label=unique_text[i][0],
                   labelloc='b', imagepos='tc', fontsize="15", height=ideal_height, fontcolor=unique_text[i][1])

    for i in range(len(edge_row_indices)):
        start = unique_text[edge_row_indices[i]][0]
        end = unique_text[edge_col_indices[i]][0]
        distance = nonzero_value_list[0, i] + 1.3
        distance = (distance ** 2) * 7.5
        G.add_edge(start, end, len=abs(distance))

    G.draw('english/english_mst.pdf',
           format='pdf', prog='neato')


if __name__ == "__main__":
    main()
