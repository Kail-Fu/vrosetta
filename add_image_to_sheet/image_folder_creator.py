import urllib.request
import pandas as pd
import requests


reference = pd.read_csv(
    '/Users/fkl/vr/dataset/final_reordered_4454.csv')


def download_image(url, file_name, headers):
    # Send GET request
    response = requests.get(url, headers=headers)
    # Save the image
    if response.status_code == 200:
        with open(file_name, "wb") as f:
            f.write(response.content)
    else:
        print(response.status_code)


# print(reference["Link to Image"])
headers = {
    "User-Agent": "Chrome/51.0.2704.103",
}
for _, row in reference.iterrows():
    link = row["Link to Image"]
    image_name = '/Users/fkl/vr/dataset/web_low_res/'+link.split("/")[-1]
    download_image(link, image_name, headers)
    # link = urllib.parse.quote(link)  # <- here
    # urllib.request.urlretrieve(
    #     link, '/Users/fkl/vr/dataset/web_low_res/' + image_name)
