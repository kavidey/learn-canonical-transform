# %%
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# from multiprocessing.dummy import Pool
from multiprocessing import Pool
from functools import partial
# %%
d_thresh = 1
e_thresh = 0.6
# %%
npy_dir = Path("npy/")
files = list(npy_dir.glob("*.npz"))
# %%
def process_sim(d_thresh, e_thresh, f):
    sim = np.load(f)

    unstable = np.bitwise_or(sim['d'][0] > d_thresh, sim['e'][0] > e_thresh)

    unstable_time = np.nonzero(unstable)[0]
    # print(unstable.sum())
    # print(sim['d'][0].max(), sim['e'].max())
    if unstable_time.size > 0:
        # unstable_time = unstable_time[0]
        unstable_time = sim['time'][unstable_time[0]]
    else:
        unstable_time = -1
    
    return f.stem.split('_')[1], unstable_time, np.max(sim['d'][0]), sim['e'][0,-1]#np.max(sim['e'][0])

# change this number to make sure you are not hitting disk-io bottlenecks
with Pool(20) as p:
    results = list(tqdm(p.imap(partial(process_sim, d_thresh, e_thresh), files), total=len(files)))
# %%
df = pd.DataFrame(results, columns=['id', 'unstable_time', 'max_d', 'max_e'])

df['unstable_time'].max()
# %%
# get the ids of the highest eccentricity sims:
ids_orig = df.sort_values(by=["max_e"]).tail(75)['id'].tolist()
ids = map(lambda i: i.replace('p','+'), ids_orig)
ids = map(lambda i: i.replace('m','-'), ids)
ids = np.array(list(map(int, ids)), dtype=int)
ids = ids // 2 + 500
ids
# 403,19,354,140,926,167,155,701,424,842,202,40,3,380,722,444,157,634,350,355,287,962,670,558,775,573,990,13,349,790,357,191,605,677,836,624,75,0,389,458,281,26,713,416,601,79,382,893,767,546,179,830,314,278,916,829,80,299,237,73,269,651,539,951,21,757,372,663,312,123,47,356,521,370,599
# %%
npy_f = npy_dir/(files[20].stem+".npz")
sim = np.load(npy_f)
# %%
import matplotlib.pyplot as plt

E0 = sim["Energy"][0]
Eerr = np.abs((E0 - sim["Energy"]) / E0)
plt.plot(sim['time']/(np.pi*2), Eerr)
plt.yscale("log")
plt.xscale("log")
# %%
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
for f in files:
    npy_f = npy_dir/(f.stem+".npz")
    sim = np.load(npy_f)

    axs[0].plot(sim['time']/(np.pi*2), sim['e'][0], linewidth=0.1)
    axs[0].set_ylabel("Eccentricity")
    axs[0].set_xlabel("Time [yr]")

    E0 = sim["Energy"][0]
    Eerr = np.abs((E0 - sim["Energy"]) / E0)
    axs[1].plot(sim['time']/(np.pi*2), Eerr)
    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Time [yr]")
    axs[1].set_ylabel("Energy Error")
# %%
fig, axs = plt.subplots(1, 2, figsize=(16, 5))

for simid in df[df['max_e'] > 0.6]['id']:
    npy_f = npy_dir/(f"solarsystem_{simid}.npz")
    sim = np.load(npy_f)

    axs[0].plot(sim['time']/(np.pi*2), sim['e'][0], linewidth=0.1)
    axs[0].set_ylabel("Eccentricity")
    axs[0].set_xlabel("Time [yr]")

    E0 = sim["Energy"][0]
    Eerr = np.abs((E0 - sim["Energy"]) / E0)
    axs[1].plot(sim['time']/(np.pi*2), Eerr)
    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Time [yr]")
    axs[1].set_ylabel("Energy Error")
# %%
