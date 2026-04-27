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
for i, sim in enumerate(e10_sims):
    plt.plot(sim['time']*action_angle_tools.TO_YEAR/1e9, sim['e'][0], linewidth=0.2, color='lightgrey', rasterized=True, label=r'$e \sim 1.0$ sims' if i==0 else None)
for sim in e05_sims:
    plt.plot(sim['time']*action_angle_tools.TO_YEAR/1e9, sim['e'][0], linewidth=0.2, rasterized=True)
plt.xlim(0,5)
plt.ylim(0,1)
plt.xlabel("Gyr")
plt.ylabel("$e$")
plt.legend()

plt.tight_layout()
plt.savefig("figs/e05-batch.eps", dpi=450)
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
# %%
plt.figure(figsize=(5,3))
plt.errorbar([0.5, 0.7, 0.9, 1.0], np.array(instability_rate)*100, yerr=np.array(ci95)*100, capsize=3, fmt="r--o", ecolor = "black")
plt.xlabel("$e$ reduction")
plt.ylabel("Instability %")

plt.tight_layout()
plt.savefig("figs/instability-rate.eps")
# %%
