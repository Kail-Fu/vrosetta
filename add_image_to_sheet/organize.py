import pandas as pd
import os
df = pd.ExcelFile(
    '/Users/fkl/vr/dataset/Final_Compiled Captions, Chars and Links Included_Updated 20221104.xlsx').parse("Sheet1")
# x = []
# x.append(df['Link to Image'])
# x = x[0]
# seen = set()
# dupes = [a for a in x if a in seen or seen.add(a)]
# print(len(x), len(dupes), len(seen))
# # print(dupes)
# nan_in_col = df[df['Link to Image'].isna()]
dir_list = os.listdir("./new_images")
new_list = []
for d in dir_list:
    new_list.append((int(d.split("_")[
                    0]), "https://cs.brown.edu/research/vis/tgurth/vrosetta/Compiled_Images_Lower_Res/"+d))
new_list.sort()
f = open("myfile.txt", "w")
for d in new_list:
    f.write(d[1])
    f.write("\n")
f.close


# # count = 0
# for idx, row in df.iterrows():
#     if type(row["Link to Image"]) != str:
#         new_link = dic[row["Image Number (char_id or table_id)"]]
#         new_link = "https://cs.brown.edu/research/vis/tgurth/vrosetta/Compiled_Images_Lower_Res/"+new_link
#         print(new_link)

# print(count)
