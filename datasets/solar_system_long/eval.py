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
    
    return f.stem.split('_')[1], unstable_time, np.max(sim['d'][0]), np.max(sim['e'][0])

# change this number to make sure you are not hitting disk-io bottlenecks
with Pool(20) as p:
    results = list(tqdm(p.imap(partial(process_sim, d_thresh, e_thresh), files), total=len(files)))
# %%
df = pd.DataFrame(results, columns=['id', 'unstable_time', 'max_d', 'max_e'])

df['unstable_time'].max()
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
