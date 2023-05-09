import pandas as pd
from collections import defaultdict

# df = pd.read_csv(
#     'english/english_reordered_matrix_off_caption_vectors_xlm_r.csv', header=None)
# df = pd.read_csv(
#     'french/french_reordered_matrix_off_caption_vectors_xlm_r.csv', header=None)
# df = pd.read_csv(
#     'viet/new/viet_reordered_matrix_off_caption_vectors_distiluse2.csv', header=None)
df = pd.read_csv(
    '3_languages/2_languges_combined_reordered_matrix_off_caption_vectors.csv', header=None)
reference = pd.ExcelFile(
    '/Users/fkl/vr/dataset/Final_Compiled Captions, Chars and Links Included_Updated 20221108.xlsx').parse("Sheet1")

eng_list = ['English']
french_list = ['French Text']
norm_list = ['Char Text']
link_list = ['Link to Image']
idx_list = df.iloc[:, 0].tolist()
for i in idx_list:
    eng_list.append(reference.iloc[i]["English Translation"])
    french_list.append(reference.iloc[i]["French Text"])
    norm_list.append(reference.iloc[i]["Char Text"])
    link_list.append(reference.iloc[i]["Link to Image"])


# df.loc[-1] = eng_list
# df.loc[-1] = french_list
df.loc[-1] = norm_list
df.index = df.index + 1  # shifting index
df.sort_index(inplace=True)
df = df.drop(df.columns[[0]], axis=1)
# df.insert(loc=0, column=0, value=eng_list)
# df.insert(loc=1, column=-1, value=french_list)
# df.insert(loc=0, column=0, value=french_list)
# df.insert(loc=1, column=-1, value=eng_list)
df.insert(loc=0, column=0, value=norm_list)
df.insert(loc=1, column=-1, value=eng_list)
df.insert(loc=2, column=-2, value=french_list)
df.insert(loc=3, column=-3, value=link_list)
# df.to_csv('viet/new/viet_reordered_4454_distiluse2.csv', encoding='utf-8-sig')
# df.to_csv('french/french_reordered_4454_xlm_r.csv', encoding='utf-8-sig')
# df.to_csv('english/english_reordered_4454_xlm_r.csv', encoding='utf-8-sig')
df.to_csv('3_languages/2_languages_4454.csv', encoding='utf-8-sig')


# # french_dic = defaultdict(str)
# # norm_dic = defaultdict(str)
# # link_dic = defaultdict(str)
# # for _, row in reference.iterrows():
# #     french_dic[row["English Translation"]] = row["French Text"]
# #     norm_dic[row["English Translation"]] = row["Char Text"]
# #     link_dic[row["English Translation"]] = row["Link to Image"]
# # french_col = ["French Text"]
# # norm_col = ["Viet Translation"]
# # link_col = ["Link to Image"]
# for e in eng_order:
#     french_col.append(french_dic[e])
#     norm_col.append(norm_dic[e])
#     link_col.append(link_dic[e])

# df.insert(loc=1, column=-1, value=french_col)
# df.insert(loc=2, column=-2, value=norm_col)
# df.insert(loc=3, column=-3, value=link_col)

# df.to_csv("final_reordered_4454.csv")
