# vrosetta
Code for building Virtual Rosetta

## How to use
1. Download dataset from Virtual Rosetta Google Drive https://drive.google.com/file/d/1sWXpKnDpAiD2aaXLEK_lKNZfbPJVs2i-/view?usp=sharing
2. Move the files in this folder to /henri_oger/view_page
3. to run any of the code, just enter "python [file_name]"
4. Specifically, if you are interested to create MST based on one image. Download dataset sheet from https://docs.google.com/spreadsheets/d/1fMK7w7XBklu9-qsAatP2F2bFTG5CTSI9/edit?usp=share_link&ouid=107527649313550538089&rtpof=true&sd=true. Go to mst folder. You should first run "similarity_gen.py" to get the similarity matrix. Then run "matrix_reorder.py" to match each caption with its associating information. Then run "ordered_sheet_modification.py" to add associated information to the caption row in a new sheet. Finally, run "mst_off_csv.py" to get MST. (Use "G = PG.AGraph()" to produce overlapping MST. Use "G = PG.AGraph(overlap="prism")" to produce non-overlapping one)
5. If you want to combine all three languages' embeddings together to create MST, do all things same as the last bullet point except replaing running "similarity_gen.py" with running "3language_matrix_gen.py"
6. If you want to generate dimension reduction graph for the dataset, use "tsne_generator.py" and "umap_generator.py" in the dimension_reduction folder.

## How each file works
* convert_view (by Anessa): a practice to generate a wall of images (to display at the museum)

* convert_view2 (by Anessa): 
1. generate similarity matrix using average_word_embeddings_komninos
2. evaluate matrix by top similar words, kmeans, and h-clustering

* kail_similarity (by Kail): 
1. You can choose to transfer words to vectors by average_word_embeddings_komninos, distilbert-base-nli-stsb-mean-tokens, sentence-transformers/all-mpnet-base-v2, or sentence-transformers/multi-qa-mpnet-base-dot-v1 (all of those are pre-trained BERT model.) and then calculate cos similarity.
2. Use it to generate 'json_data_for_directed_force_graph.json', which can be used to feed into d3 force directed graph here: https://observablehq.com/@64127e29fecf3d15/distilbert-ward-top-5-connections
3. Cluster by kmeans
4. Generate different h-clustering. For example, "linkage(selected_embedding, 'single')" corresponds to single method, "linkage(selected_embedding, 'ward')" corresponds ward method
5. write similarity matrix "kail_similarity.csv"
6. List most and least similar captions for any given caption according to the selected similarity matrix

* kail_larger_than.py: generate json object to feed into d3. result: https://observablehq.com/@64127e29fecf3d15/force-directed-graph-threshold-0-75

* kail_top_2_bot_1_trail.py: generate json object to feed into d3. result: https://observablehq.com/@64127e29fecf3d15/distilbert-ward-top-2-last-1-connections

* tree_kail.py: generate MST according to cosine similarity

* add_image_to_sheet/image_folder_creator.py: create a folder containing all low resolution images with the provided links from the sheet

* add_image_to_sheet/add_image.py: after creating the image folder, add each image to the sheet
 

## Common issue
* Python cannot find "...", mostly it would be an issue of configuration if "..." is something to be imported. Just use something like pip install "...". If doesn't work, search the error message.
* Cannot find similarity matrix. This is a common issue. You need to run convert_view.py first to generate matrix, then change the name to something like "kail_similarity" and rerun the previous code.
