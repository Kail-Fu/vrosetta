import pandas as pd
import numpy as np
from collections import defaultdict
import math


def scaler(df, target_mean):
    def transformed_mean(power, df):
        transformed_df = df.pow(power)
        return transformed_df.mean().mean()

    # Define the function for which we want to find the root (f(p) = transformed_mean(p) - 0.5)
    def f(power, df):
        return transformed_mean(power, df) - target_mean
    # Use the bisection method to find the power p
    min_power = 0.1
    max_power = 4
    tolerance = 1e-6
    max_iterations = 1000

    for _ in range(max_iterations):
        mid_power = (min_power + max_power) / 2
        if f(mid_power, df) * f(min_power, df) < 0:
            max_power = mid_power
        else:
            min_power = mid_power

        if max_power - min_power < tolerance:
            power = mid_power
            break
    # Apply the found power to the DataFrame
    return df.pow(power)


df_viet = pd.read_csv(
    'viet/new/viet_similarity_matrix_viet_sbert.csv', header=None)
df_viet = df_viet.iloc[1:]
del df_viet[df_viet.columns[0]]
df_viet = scaler(df_viet.astype(float).clip(lower=0), 0.5)
# df_viet.to_csv(
#     "3_languages/scaled_viet.csv", encoding='utf-8-sig')

df_french = pd.read_csv(
    'french/french_similarity_matrix_pm_mpnet.csv', header=None)
df_french = df_french.iloc[1:]
del df_french[df_french.columns[0]]
df_french = scaler(df_french.astype(float).clip(lower=0), 0.5)
df_french.to_csv(
    "3_languages/scaled_french.csv", encoding='utf-8-sig')

# df_english = pd.read_csv(
#     'english/english_similarity_matrix_pm_mpnet.csv', header=None)
# df_english = df_english.iloc[1:]
# del df_english[df_english.columns[0]]
# df_english = scaler(df_english.astype(float).clip(lower=0), 0.5)

print(df_viet.mean().mean(), df_french.mean().mean())

combined_matrix = df_viet + df_french
combined_matrix = 2 - combined_matrix

# combined_matrix = pd.DataFrame(np.minimum(df_viet, df_french))
# combined_matrix = 1 - combined_matrix


def convert_close_to_zero(value, threshold=1e-4):
    if abs(value) < threshold:
        return 0
    return value


combined_matrix = combined_matrix.applymap(lambda x: convert_close_to_zero(x))
combined_matrix.reset_index(drop=True, inplace=True)
combined_matrix.insert(0, 'Index', combined_matrix.index)

combined_matrix.to_csv(
    "3_languages/2_languges_combined_reordered_matrix_off_caption_vectors.csv", encoding='utf-8-sig')
# combined_matrix.to_csv(
#     "3_languages/minimum_3_languages_reordered_matrix_off_caption_vectors.csv", encoding='utf-8-sig')

# df_viet = (df_viet-df_viet.mean())/df_viet.std()
# df_viet = (df_viet-df_viet.min())/(df_viet.max()-df_viet.min())

# df_viet.to_csv('normalized.csv', encoding='utf-8-sig')
