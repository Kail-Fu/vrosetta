import pandas as pd

english_df = pd.read_csv(
    'english/english_reordered_matrix_off_caption_vectors_pm_mpnet.csv', header=None)
french_df = pd.read_csv(
    'french/french_reordered_matrix_off_caption_vectors_pm_mpnet.csv', header=None)
viet_df = pd.read_csv(
    'viet/viet_reordered_matrix_off_caption_vectors_viet_sbert.csv', header=None)
reference = pd.ExcelFile(
    '/Users/fkl/vr/dataset/Final_Compiled Captions, Chars and Links Included_Updated 20221108.xlsx').parse("Sheet1")

eng_list = ['English']
french_list = ['French Text']
norm_list = ['Char Text']
link_list = ['Link to Image']
idx_list = english_df.iloc[:, 0].tolist()
for i in idx_list:
    eng_list.append(reference.iloc[i]["English Translation"])
    french_list.append(reference.iloc[i]["French Text"])
    norm_list.append(reference.iloc[i]["Char Text"])
    link_list.append(reference.iloc[i]["Link to Image"])

# english
english_df.loc[-1] = eng_list
english_df.index = english_df.index + 1  # shifting index
english_df.sort_index(inplace=True)
english_df = english_df.drop(english_df.columns[[0]], axis=1)
english_df.insert(loc=0, column=0, value=eng_list)
english_df.insert(loc=1, column=-1, value=french_list)
english_df.insert(loc=2, column=-2, value=norm_list)
english_df.insert(loc=3, column=-3, value=link_list)
english_df.to_csv("english/english_4454.csv", encoding='utf-8-sig', index=False, header=False)

# french
french_df.loc[-1] = french_list
french_df.index = french_df.index + 1  # shifting index
french_df.sort_index(inplace=True)
french_df = french_df.drop(french_df.columns[[0]], axis=1)
french_df.insert(loc=0, column=0, value=french_list)
french_df.insert(loc=1, column=-1, value=eng_list)
french_df.insert(loc=2, column=-2, value=norm_list)
french_df.insert(loc=3, column=-3, value=link_list)
french_df.to_csv('french/french_4454.csv', encoding='utf-8-sig', index=False, header=False)

# viet
viet_df.loc[-1] = norm_list
viet_df.index = viet_df.index + 1  # shifting index
viet_df.sort_index(inplace=True)
viet_df = viet_df.drop(viet_df.columns[[0]], axis=1)
viet_df.insert(loc=0, column=0, value=norm_list)
viet_df.insert(loc=1, column=-1, value=eng_list)
viet_df.insert(loc=2, column=-2, value=french_list)
viet_df.insert(loc=3, column=-3, value=link_list)
viet_df.to_csv('viet/viet_4454.csv', encoding='utf-8-sig', index=False, header=False)