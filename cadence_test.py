# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import rebound as rb
from utils import get_simarchive_integration_results
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
# %%
def load_sim(path, filter_freq=None):
    results = get_simarchive_integration_results(str(path), coordinates='heliocentric')
    
    m = masses[..., None].repeat(results['a'].shape[1], axis=-1)
    G = 1
    beta = ((1 * m) / (1 + m))
    mu = G * (1 + m)
    results['Lambda'] = beta * np.sqrt(mu * results['a'])
    
    M = results['l'] - results['pomega']
    results['lambda'] = M + results['pomega']

    results['x'] = np.sqrt(results['Lambda']) * np.sqrt(1 - np.sqrt(1-results['e']**2)) * np.exp(1j * results['pomega'])
    results['y'] = np.sqrt(2 * results['Lambda']) * np.power(1-results['e']**2, 1/4) * np.sin(results['inc']/2) * np.exp(1j * results['Omega'])

    # coordinate pairs are:
    # - Lambda, Lambda
    # - x, -i * x_bar
    # - y, -i * y_bar

    fs_arcsec_per_yr = (TO_ARCSEC_PER_YEAR / np.gradient(results['time']).mean()) * 2 * np.pi

    if filter_freq:
        b, a = scipy.signal.butter(10, filter_freq, 'low', fs=fs_arcsec_per_yr) # type: ignore[reportUnknownVariableType]
        results['x'] = scipy.signal.lfilter(b, a, results['x'])
        results['y'] = scipy.signal.lfilter(b, a, results['y'])

        for key in results.keys():
            results[key] = results[key][..., 100:]

    return results
# %%
dataset_path = Path("datasets/cadence_test")
rb_sim = rb.Simulation(str(dataset_path/'planets.bin'))
masses = np.array(list(map(lambda ps: ps.m, rb_sim.particles))[1:], dtype=np.float64)
TO_ARCSEC_PER_YEAR = 60*60*180/np.pi * (2*np.pi)
TO_YEAR = 1/(2*np.pi)
# %%
sims = {}
# for f in dataset_path.glob("solarsystem_p0.*.bin"):
#     sims[int(float(f.stem.split('.')[-1]))] = load_sim(f)
sims['test'] = load_sim(dataset_path/'solarsystem_p0.bin')
# sims['test512'] = load_sim(dataset_path/'solarsystem_p0.avx512.bin')
# sims['test512gr'] = load_sim(dataset_path/'solarsystem_p0.avx512.gr.bin')
sims['test512gr'] = load_sim(dataset_path/'solarsystem_m1000.avx512.gr.bin')
# %%
for cadence, sim in sims.items():
    print("#"*20)
    print("SIM CADENCE:", cadence)
    fs_arcsec_per_yr = (TO_ARCSEC_PER_YEAR / np.gradient(sim['time']).mean()) * 2 * np.pi
    print("sample rate (\"/yr):", fs_arcsec_per_yr)
    print("dt (yr):", np.gradient(sim['time']).mean() / (2 * np.pi))
    print("total samples:", sim['time'].shape[0])
    print("\n")
# %%
planets = ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")
for cadence, sim in sims.items():
    print("#"*20)
    print("SIM CADENCE:", cadence)
    planet_ecc_fmft = dict()
    planet_inc_fmft = dict()
    for i,pl in enumerate(planets):
        planet_ecc_fmft[pl] = fmft(sim['time'][:10001],sim['x'][i][:10001],14)
        planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))
        planet_e_freqs_arcsec_per_yr = planet_e_freqs * TO_ARCSEC_PER_YEAR

        planet_inc_fmft[pl] = fmft(sim['time'][:10001],sim['y'][i][:10001],14)
        planet_inc_freqs = np.array(list(planet_inc_fmft[pl].keys()))
        planet_inc_freqs_arcsec_per_yr = planet_inc_freqs * TO_ARCSEC_PER_YEAR

        print("")
        print(pl)
        print("g")
        print("-------")
        for g in planet_e_freqs:
            print("{:+07.3f} \t {:0.6f}".format(g * TO_ARCSEC_PER_YEAR, np.abs(planet_ecc_fmft[pl][g])))
        print("s")
        print("-------")
        for s in planet_inc_freqs:
            print("{:+07.3f} \t {:0.6f}".format(s * TO_ARCSEC_PER_YEAR, np.abs(planet_inc_fmft[pl][s])))

    print("\n\n\n")
# %%
plt.plot(np.real(sims['test']['x'][5][:10001]), np.imag(sims['test']['x'][5][:10001]))
plt.plot(np.real(sims['test512gr']['x'][5][:10001]), np.imag(sims['test512gr']['x'][5][:10001]))
# %%
plt.plot(np.real(sims['test']['e'][5][:100]))
plt.plot(np.real(sims['test512gr']['e'][5][:100]))
# %%
# print(np.allclose(sims['test']['x'], sims['test512']['x'][:,:-1]))
print(np.allclose(sims['test']['x'], sims['test512gr']['x']))
# %%
