# %%
import requests
import urllib.request
import time
from tqdm import tqdm
# %%
response = requests.get("https://zenodo.org/api/records/4299102").json()
files = response['files']
# %%
for f in tqdm(files):
    url = f['links']['self']
    urllib.request.urlretrieve(url, url.split('/')[-2])
    time.sleep(30)