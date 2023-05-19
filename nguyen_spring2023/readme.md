# Course Material
This folder aims to enable students to visualize a historical dataset

## Preparation
1. **Download Sheet**
Follow this [link](https://docs.google.com/spreadsheets/d/16yMkE7Mq18QrLiFVpHl8XWvVdxIKrZy8/edit?usp=sharing&ouid=107527649313550538089&rtpof=true&sd=true) to download the sheet containing 4454 rows data with captions in 3 languages and other properties.

2. **Download Images**
Follow this [link](https://drive.google.com/file/d/10gt402_PHWq2Q_trCCoWJQYXRksBGYBg/view?usp=sharing) to download the zip of those captions' corresponding images. Uncompress it and put it in this folder.

3. **Fullfill Env Requirements**
Install all necessary requirements
```bash
pip install -r requirements.txt
```

4. **Generate Relational Data**
First, go to "similarity_gen.py" and change "FINAL_COMPLIED_SHEET" to the path of the sheet you just downloaded. Then, run 
```bash
python3 similarity_gen.py
```
You will then get similarity_matrix (versions with caption and without captions) in 3 languages' folders.
Next, run the following command. This could take a while (~10 minutes on 2021 Macbook Pro)
```bash
python3 matrix_reorder.py
```
Then, go to "ordered_sheet_modification.py" to change "reference" to the path of the sheet you just downloaded (same as "FINAL_COMPLIED_SHEET"). And run
```bash
python3 ordered_sheet_modification.py
```

## Visualize in Minimum Spanning Tree
Run 
```bash
python3 mst_off_csv.py
```

## Visualize in Hierarchical Clustering 
Run 
```bash
python3 clustering.py
```

## Visualize in Radial Spanning Tree 
Run 
```bash
python3 radial_json_generator.py
```
You could change the word you want to put in the center by changing "keyword" variable. After running, it will generate a file called "radial.json". Go to this [page](https://observablehq.com/d/c7cbeabbeffbc1c2), click the files button on the top right, and click the switch button. Then, upload the "radial.json" you just generated. It should reload according to the json.
