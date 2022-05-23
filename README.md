# vrosetta
Code for building Virtual Rosetta

## How to use
1. Download dataset from Virtual Rosetta Google Drive https://drive.google.com/file/d/1sWXpKnDpAiD2aaXLEK_lKNZfbPJVs2i-/view?usp=sharing
2. Move the files in this folder to /henri_oger/view_page
3. to run any of the code, just enter "python [file_name]"

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

## Common issue
* Python cannot find "...", mostly it would be an issue of configuration if "..." is something to be imported. Just use something like pip install "...". If doesn't work, search the error message.
* Cannot find similarity matrix. This is a common issue. You need to run convert_view.py first to generate matrix, then change the name to something like "kail_similarity" and rerun the previous code.
