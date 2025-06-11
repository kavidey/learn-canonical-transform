# %%
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jnr
import numpy as np
import matplotlib.pyplot as plt

from flax import nnx
import optax
import orbax.checkpoint as ocp

import rebound as rb

from celmech.nbody_simulation_utilities import get_simarchive_integration_results
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
# %%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
# %%
dataset_path = Path('datasets') / 'asteroid_integration' / 'outer_planets'
base_desn = '00001'
TO_ARCSEC_PER_YEAR = 60*60*180/np.pi * (2*np.pi)
# %%
sims = list(dataset_path.glob('*.sa'))
desns = list(map(lambda p: p.stem.split('_')[-1], sims))
# %%
sim = rb.Simulation(str(dataset_path/'planets.bin'))
masses = np.array(list(map(lambda ps: ps.m, sim.particles))[1:] + [1])

def load_sim(desn):
    results = get_simarchive_integration_results(str(dataset_path / f"{'_'.join(sims[0].stem.split('_')[:-1])}_{desn}.sa"), coordinates='heliocentric')
    
    results['X'] = np.sqrt(2*(1-np.sqrt(1-results['e']**2))) * np.exp(1j * results['pomega'])
    results['Y'] = (1-results['e']**2)**(0.25) * np.sin(0.5 * results['inc'] ) * np.exp(1j * results['Omega']) 

    results['G'] = np.sqrt(masses)[..., None].repeat(results['X'].shape[1], axis=-1) * np.power(sim.G * sim.particles[0].m * results['a'], 1/4) * results['X']

    return results
# %%
base_sim = load_sim(base_desn)

planets = ("Jupiter","Saturn","Uranus","Neptune")
planet_ecc_fmft = dict()
planet_inc_fmft = dict()
for i,pl in enumerate(planets):
    planet_ecc_fmft[pl] = fmft(base_sim['time'],base_sim['G'][i],14)
    planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))
    planet_e_freqs_arcsec_per_yr = planet_e_freqs * TO_ARCSEC_PER_YEAR

    planet_inc_fmft[pl] = fmft(base_sim['time'],base_sim['Y'][i],8)
    planet_inc_freqs = np.array(list(planet_inc_fmft[pl].keys()))
    planet_inc_freqs_arcsec_per_yr = planet_inc_freqs * TO_ARCSEC_PER_YEAR

    print("")
    print(pl)
    print("g")
    print("-------")
    for g in planet_e_freqs:#[:6]:
        print("{:+07.3f} \t {:0.6f}".format(g * TO_ARCSEC_PER_YEAR, np.abs(planet_ecc_fmft[pl][g])))
    print("s")
    print("-------")
    for s in planet_inc_freqs:#[:4]:
        print("{:+07.3f} \t {:0.6f}".format(s * TO_ARCSEC_PER_YEAR, np.abs(planet_inc_fmft[pl][s])))
# %%
g_vec = np.zeros(4)
s_vec = np.zeros(4)

g_vec[:3] = np.array(list(planet_ecc_fmft['Jupiter'].keys()))[:3]
g_vec[3] = list(planet_ecc_fmft['Neptune'].keys())[0]
s_vec[0] = list(planet_inc_fmft['Jupiter'].keys())[0]
s_vec[1] = list(planet_inc_fmft['Jupiter'].keys())[1]
s_vec[2] = list(planet_inc_fmft['Jupiter'].keys())[2]
s_vec[3] = list(planet_inc_fmft['Jupiter'].keys())[3]

omega_vec = np.concatenate((g_vec,s_vec))
g_and_s_arc_sec_per_yr = omega_vec * TO_ARCSEC_PER_YEAR
with np.printoptions(suppress=True, precision=3):
    print(g_and_s_arc_sec_per_yr)
# %%
ecc_rotation_matrix_T_base = np.zeros((5, 5))

# planet ecc in terms of planet modes
mode_angle = [0, 0, 0, 0, 0]
for i, pl in enumerate(planets):
    planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))
    for j, g in enumerate(g_vec):
        found_g = find_nearest(planet_e_freqs, g)
        if np.abs((found_g - g)/g) > 0.1:
            continue
        ecc_rotation_matrix_T_base[i][j] = np.abs(planet_ecc_fmft[pl][found_g])
        if mode_angle[i] == 0:
            mode_angle[i] = np.angle(planet_ecc_fmft[pl][found_g])
        if mode_angle[i] != 0 and np.abs(mode_angle[i] - np.angle(planet_ecc_fmft[pl][found_g])) > np.pi/2:
            ecc_rotation_matrix_T_base[i][j] *= -1

inc_rotation_matrix_T_base = np.zeros((5, 5))
for i, pl in enumerate(planets):
    planet_i_freqs = np.array(list(planet_inc_fmft[pl].keys()))
    for j, s in enumerate(s_vec):
        found_s = find_nearest(planet_i_freqs, s)
        if np.abs((found_s - s)/s) > 0.1:
            continue
        inc_rotation_matrix_T_base[i][j] = np.abs(planet_inc_fmft[pl][found_s])
# %%
for desn in desns:
    # sim = load_sim(desn)
    # asteroid_ecc_fmft = fmft(sim['time'],sim['G'][-1],14)
    # asteroid_inc_fmft = fmft(sim['time'],sim['Y'][-1],8)

    ecc_rotation_matrix_T = ecc_rotation_matrix_T_base.copy()
    inc_rotation_matrix_T = inc_rotation_matrix_T_base.copy()

    asteroid_e_freqs = np.array(list(asteroid_ecc_fmft.keys()))
    asteroid_i_freqs = np.array(list(asteroid_inc_fmft.keys()))
    print("Asteroid\ng\n--------")
    for g in asteroid_e_freqs:#[:6]:
        print("{:+07.3f} \t {:0.6f}".format(g * TO_ARCSEC_PER_YEAR, np.abs(asteroid_ecc_fmft[g])))
    print('\ns\n--------')
    for s in asteroid_i_freqs:#[:4]:
        print("{:+07.3f} \t {:0.6f}".format(s * TO_ARCSEC_PER_YEAR, np.abs(asteroid_inc_fmft[s])))
    print()
    ## ECC ###
    # asteroid ecc in terms of planet modes
    for j, g in enumerate(g_vec):
        found_g = find_nearest(asteroid_e_freqs, g)
        if np.abs((found_g - g)/g) > 0.1:
            continue
        ecc_rotation_matrix_T[4][j] = np.abs(asteroid_ecc_fmft[found_g])

    # find the largest mode that isn't in g_vec, thats the asteroid mode
    for g in asteroid_e_freqs:
        found_g = find_nearest(g_vec, g)
        if np.abs((found_g - g)/g) > 0.1:
            asteroid_g = g
            break
    
    # asteroid ecc in terms of its own mode
    ecc_rotation_matrix_T[4][4] = np.abs(asteroid_ecc_fmft[g])
    
    ### INC ###
    # asteroid inc in terms of planet modes
    for j, s in enumerate(s_vec):
        found_s = find_nearest(asteroid_i_freqs, s)
        if np.abs((found_s - s)/s) > 0.1:
            continue
        inc_rotation_matrix_T[4][j] = np.abs(asteroid_inc_fmft[found_s])

    # find the largest mode that isn't in g_vec, thats the asteroid mode
    for s in asteroid_i_freqs:
        found_s = find_nearest(s_vec, s)
        if np.abs((found_s - s)/s) > 0.1:
            asteroid_s = s
            break
    
    # asteroid ecc in terms of its own mode
    inc_rotation_matrix_T[4][4] = np.abs(asteroid_inc_fmft[s])
    break

# normalize so determinant is 1
# ecc_rotation_matrix_T = ecc_rotation_matrix_T * 1 / jnp.pow(np.linalg.det(ecc_rotation_matrix_T), 1/5)
# inc_rotation_matrix_T = inc_rotation_matrix_T * 1 / jnp.pow(np.linalg.det(inc_rotation_matrix_T), 1/5)

with np.printoptions(suppress=True, precision=4):
    print("ecc")
    print(ecc_rotation_matrix_T)
    print("inc")
    print(inc_rotation_matrix_T)
# %%
ecc_rotation_matrix_T = ecc_rotation_matrix_T[:4,:4]
# ecc_rotation_matrix_T = ecc_rotation_matrix_T / np.linalg.norm(ecc_rotation_matrix_T, axis=1)[...,None].repeat(4, axis=1)
normalize_cols = 2 / np.sum(np.abs(ecc_rotation_matrix_T), axis=0)
ecc_rotation_matrix_T = ecc_rotation_matrix_T * normalize_cols
print(np.linalg.det(ecc_rotation_matrix_T))

npts = 100_000
skip_pts = 100
# Phi = (np.linalg.inv(ecc_rotation_matrix_T) @ sim['G'][:4])
Phi = (inc_rotation_matrix_T.T @ sim['G'])

fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i in range(4):
    pts = sim['G'][i][:npts][::skip_pts]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Phi[i][:npts][::skip_pts]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
transformedY = (sim['Y'].T @ inc_rotation_matrix_T.T).T

fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i in range(5):
    pts = sim['Y'][i][:npts][::skip_pts]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = transformedY[i][:npts][::skip_pts]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
