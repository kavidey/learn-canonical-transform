# %%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats

import action_angle_tools

%config InlineBackend.figure_format = 'retina'
# %%
e05_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_e0.5' / 'npy'
e07_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_e0.7' / 'npy'
e09_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_e0.9' / 'npy'
e10_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_orig' / 'npy'
# %%
e05_sims = [np.load(s, allow_pickle=True)['arr_0'][()] for s in e05_dataset.glob("*[!hires].npz")]
e07_sims = [np.load(s, allow_pickle=True)['arr_0'][()] for s in e07_dataset.glob("*[!hires].npz")]
e09_sims = [np.load(s, allow_pickle=True)['arr_0'][()] for s in e09_dataset.glob("*[!hires].npz")]
e10_sims = [np.load(s, allow_pickle=True)['arr_0'][()] for s in e10_dataset.glob("*[!hires].npz")]
# %%
plt.figure(figsize=(5,3))
for sim in e10_sims:
    plt.plot(sim['time']*action_angle_tools.TO_YEAR/1e9, sim['e'][0], linewidth=0.2, rasterized=True)
plt.xlim(0,5)
plt.ylim(0,1)
plt.xlabel("Gyr")
plt.ylabel("$e$")

plt.tight_layout()
plt.savefig("figs/e10-batch.eps", dpi=450)
# %%
plt.figure(figsize=(5,3))
for sim in e09_sims:
    plt.plot(sim['time']*action_angle_tools.TO_YEAR/1e9, sim['e'][0], linewidth=0.2, rasterized=True)
plt.xlim(0,5)
plt.ylim(0,1)
plt.xlabel("Gyr")
plt.ylabel("$e$")

plt.tight_layout()
plt.savefig("figs/e09-batch.eps", dpi=450)
# %%
plt.figure(figsize=(5,3))
for sim in e07_sims:
    plt.plot(sim['time']*action_angle_tools.TO_YEAR/1e9, sim['e'][0], linewidth=0.2, rasterized=True)
plt.xlim(0,5)
plt.ylim(0,1)
plt.xlabel("Gyr")
plt.ylabel("$e$")

plt.tight_layout()
plt.savefig("figs/e07-batch.eps", dpi=450)
# %%
plt.figure(figsize=(5,2.5))
for i, sim in enumerate(e10_sims):
    plt.plot(sim['time']*action_angle_tools.TO_YEAR/1e9, sim['e'][0], linewidth=0.2, color='lightgrey', rasterized=True, label=r'$e \sim 1.0$ sims' if i==0 else None)
for sim in e05_sims:
    plt.plot(sim['time']*action_angle_tools.TO_YEAR/1e9, sim['e'][0], linewidth=0.2, rasterized=True)
plt.xlim(0,5)
plt.ylim(0,1)
plt.xlabel("Gyr")
plt.ylabel("$e$")

leg = plt.legend()
for line in leg.get_lines():
    line.set_linewidth(2.0)

plt.tight_layout()
plt.savefig("figs/e05-batch.eps", dpi=450)
# %%
import matplotlib as mpl

cmap = mpl.colormaps['magma']
colors = cmap(np.linspace(0.1, 0.8, 4))

skipi = 1 # increase this for faster plotting

plt.figure(figsize=(4,3))

with plt.rc_context({"lines.linewidth":0.2}):
    for i, sim in enumerate(e10_sims[::skipi]):
        plt.plot(sim['time']*action_angle_tools.TO_YEAR/1e9, sim['e'][0], color=colors[0], rasterized=True, label=r'$1.0$' if i==0 else None)
    for i, sim in enumerate(e09_sims[::skipi]):
        plt.plot(sim['time']*action_angle_tools.TO_YEAR/1e9, sim['e'][0], color=colors[1], rasterized=True, label=r'$0.9$' if i==0 else None)
    for i, sim in enumerate(e07_sims[::skipi]):
        plt.plot(sim['time']*action_angle_tools.TO_YEAR/1e9, sim['e'][0], color=colors[2], rasterized=True, label=r'$0.7$' if i==0 else None)
    for i, sim in enumerate(e05_sims[::skipi]):
        plt.plot(sim['time']*action_angle_tools.TO_YEAR/1e9, sim['e'][0], color=colors[3], rasterized=True, label=r'$0.5$' if i==0 else None)

plt.xlim(0,5)
plt.ylim(0,1)
plt.xlabel("Gyr")
plt.ylabel("$e$")

leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4)
for line in leg.get_lines():
    line.set_linewidth(2.0)

plt.tight_layout()
plt.savefig("figs/e-all-batches.eps", dpi=450)
# %%
def find_unstable(sims):
    unstable_e = []
    unstable_nan = []
    for i, sim in enumerate(sims):
        if (sim['e'][0] > 0.6).any():
            unstable_e.append(i)
        if (np.isnan(sim['e'][0]).any()):
            unstable_nan.append(i)
    return list(set(unstable_e + unstable_nan))

e05_unstable_idx = find_unstable(e05_sims)
e07_unstable_idx = find_unstable(e07_sims)
e09_unstable_idx = find_unstable(e09_sims)
e10_unstable_idx = find_unstable(e10_sims)
# %%
def calculate_instability_statistics(k, n, ci=0.95):
    assert ci == 0.95
    p_hat = k/n
    standard_error = p_hat*(1-p_hat)/n
    zstar = 1.96
    return k/n, zstar * np.sqrt(standard_error)

instability_rate, ci95 = zip(*map(
    lambda k: calculate_instability_statistics(k, 1200),
    [len(e05_unstable_idx), len(e07_unstable_idx), len(e09_unstable_idx), len(e10_unstable_idx)]
))
instability_rate = np.array(instability_rate) * 100
ci95 = np.array(ci95) * 100
# %%
ymax = ci95.copy()
ymin = ci95.copy()
ymin[0] = 0
ymax[0] = (3/1200) * 100 # use rule of 3 to estimate chance of instability happening in e~0.5

yerr = [ymin, ymax]

e = np.array([0.5, 0.7, 0.9, 1.0])
fig, ax = plt.subplots(1, 1, figsize=(4,3))
ax.errorbar(e, instability_rate, yerr=yerr, capsize=3, fmt="none", ecolor = "black", zorder=0)
ax.scatter(e, instability_rate, c=colors[::-1], zorder=10)
ax.set_xlabel("$e$ reduction")
ax.set_ylabel("Instability %")
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

plt.tight_layout()
plt.savefig("figs/instability-rate.eps")
# %%
