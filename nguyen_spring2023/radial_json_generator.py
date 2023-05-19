import csv
import json

def main():
    caption = []
    dic = {}
    max_number_of_children = 4
    threshold = 0.5
    with open('english/english_4454.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        ct = 0

        for row in reader:
            if (ct > 0):
                caption.append(row[0])
                simi_row = row[4:]

                res = sorted(range(len(simi_row)),
                             key=lambda sub: simi_row[sub])
                pt = 0
                while float(simi_row[res[pt]]) < threshold:
                    pt += 1
                res = res[:pt]
                dic[ct-1] = res
            ct += 1

    # Replace the keyword by your choice
    keyword = 'A praying monk (earthenware toy).'
    start = caption.index(keyword)
    seen = set([keyword])
    max_lvl = 4

    def to_tree(idx, lvl):
        if lvl == max_lvl:
            return {
                "name": caption[idx],
                "value": 0
            }
        else:
            childs = []
            count = 0
            for i in dic[idx]:
                if caption[i] not in seen:
                    seen.add(caption[i])
                    childs.append(to_tree(i, lvl+1))
                    count += 1
                if count == max_number_of_children:
                    break
            return {
                "name": caption[idx],
                "children": childs
            }

    radial_tree = to_tree(start, 1)
    with open("radial.json", "w") as outfile:
        json.dump(radial_tree, outfile)

if __name__ == "__main__":
    main()
