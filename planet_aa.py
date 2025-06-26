# %%
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jnr
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.optimize import minimize

import rebound as rb

from celmech.nbody_simulation_utilities import get_simarchive_integration_results
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft

import sympy

np.set_printoptions(suppress=True, precision=4)
jax.config.update("jax_enable_x64", True)
# %%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def closest_key_entry(d, target):
    """
    Given a dictionary `d` with float keys and a target float `target`,
    returns a tuple (key, value) where the key is the one in `d`
    closest to `target`.

    Parameters
    ----------
    d : dict
        Dictionary with float keys.
    target : float
        The float to compare keys against.

    Returns
    -------
    tuple
        The (key, value) pair whose key is closest to `target`.
    """
    closest_key = min(d.keys(), key=lambda k: abs(k - target))
    return closest_key, d[closest_key]
# %%
dataset_path = Path('datasets') / 'planet_integration'
TO_ARCSEC_PER_YEAR = 60*60*180/np.pi * (2*np.pi)
TO_YEAR = 1/(2*np.pi)
N = 8
# %%
rb_sim = rb.Simulation(str(dataset_path/'planets.bin'))
masses = np.array(list(map(lambda ps: ps.m, rb_sim.particles))[1:], dtype=np.float64)

def load_sim(path):
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

    return results
# %%
full_sim = load_sim(dataset_path / "planet_integration.sa")
print(full_sim['time'].shape, full_sim['time'][-1] * TO_YEAR)
# %%
keep_first = int(1e3)
sim = {}
for key, val in full_sim.items():
    sim[key] = val[..., :keep_first]
# %%
planets = ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")
planet_ecc_fmft = dict()
planet_inc_fmft = dict()
for i,pl in enumerate(planets):
    planet_ecc_fmft[pl] = fmft(sim['time'],sim['x'][i],14)
    planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))
    planet_e_freqs_arcsec_per_yr = planet_e_freqs * TO_ARCSEC_PER_YEAR

    planet_inc_fmft[pl] = fmft(sim['time'],sim['y'][i],8)
    planet_inc_freqs = np.array(list(planet_inc_fmft[pl].keys()))
    planet_inc_freqs_arcsec_per_yr = planet_inc_freqs * TO_ARCSEC_PER_YEAR

    print("")
    print(pl)
    print("g")
    print("-------")
    for g in planet_e_freqs[:6]:
        print("{:+07.3f} \t {:0.6f}".format(g * TO_ARCSEC_PER_YEAR, np.abs(planet_ecc_fmft[pl][g])))
    print("s")
    print("-------")
    for s in planet_inc_freqs[:4]:
        print("{:+07.3f} \t {:0.6f}".format(s * TO_ARCSEC_PER_YEAR, np.abs(planet_inc_fmft[pl][s])))
# %%
g_vec = np.zeros(8)
s_vec = np.zeros(8)

g_vec[0] = list(planet_ecc_fmft['Mercury'].keys())[0]
g_vec[1] = list(planet_ecc_fmft['Venus'].keys())[0]
g_vec[2] = list(planet_ecc_fmft['Earth'].keys())[3]
g_vec[3] = list(planet_ecc_fmft['Mars'].keys())[0]
g_vec[4] = list(planet_ecc_fmft['Jupiter'].keys())[0]
g_vec[5] = list(planet_ecc_fmft['Saturn'].keys())[0]
g_vec[6] = list(planet_ecc_fmft['Uranus'].keys())[1]
g_vec[7] = list(planet_ecc_fmft['Neptune'].keys())[0]

s_vec[0] = list(planet_inc_fmft['Jupiter'].keys())[0]
s_vec[1] = list(planet_inc_fmft['Jupiter'].keys())[1]
s_vec[2] = list(planet_inc_fmft['Jupiter'].keys())[2]
s_vec[3] = list(planet_inc_fmft['Jupiter'].keys())[3]

omega_vec = np.concatenate((g_vec,s_vec))
g_and_s_arc_sec_per_yr = omega_vec * TO_ARCSEC_PER_YEAR
print(g_and_s_arc_sec_per_yr)
print("should be:", [5.46, 7.34, 17.33, 18.00, 4.30, 27.77, 2.72, 0.63])
# %%
freq_thresh = 0.05
ecc_rotation_matrix_T = np.zeros((8,8))

# planet ecc in terms of planet modes
mode_angle = np.zeros(8)
for i, pl in enumerate(planets):
    planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))
    print(pl)
    for j, g in enumerate(g_vec):
        found_g = find_nearest(planet_e_freqs, g)
        with np.printoptions(suppress=True, precision=3):
            print(np.array([found_g*TO_ARCSEC_PER_YEAR, g*TO_ARCSEC_PER_YEAR, np.abs((found_g - g)/g) * 100]), np.abs((found_g - g)/g) < freq_thresh)
        if np.abs((found_g - g)/g) > freq_thresh:
            continue
        matrix_entry = np.abs(planet_ecc_fmft[pl][found_g])
        if mode_angle[i] == 0:
            mode_angle[i] = np.angle(planet_ecc_fmft[pl][found_g])
        if mode_angle[i] != 0 and np.abs(mode_angle[i] - np.angle(planet_ecc_fmft[pl][found_g])) > np.pi/2:
            matrix_entry *= -1
        ecc_rotation_matrix_T[i][j] += matrix_entry
    print()

ecc_rotation_matrix_T = ecc_rotation_matrix_T / np.linalg.norm(ecc_rotation_matrix_T, axis=0)
print(np.linalg.det(ecc_rotation_matrix_T))

print(ecc_rotation_matrix_T)

Phi = (np.linalg.inv(ecc_rotation_matrix_T) @ sim['x'])
Phi[-1] *= 1e7

fig, axs = plt.subplots(2,8,figsize=(20, 5))
for i, pl in enumerate(planets):
    axs[0][i].set_title(pl)
    pts = sim['x'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Phi[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
def objective(R, G, m):
    R = jnp.reshape(R, (8,8))
    # get a valid rotation matrix from W
    # U, S, Vh = np.linalg.svd(W)
    # R = U @ Vh

    rotation_loss = ((jnp.eye(8) - R @ R.T) ** 2).sum() #+ (np.linalg.det(R) - 1) ** 2

    Phi = R.T @ G

    J_loss = ((jnp.abs(Phi) - jnp.abs(Phi).mean(axis=1)[..., None]) ** 2).sum()

    loss = rotation_loss + J_loss
    return loss

obj_and_grad = jax.jit(jax.value_and_grad(lambda R: objective(R, sim['x'], masses)))

sol = minimize(obj_and_grad, ecc_rotation_matrix_T.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
ecc_rotation_matrix_opt_T = sol.x.reshape(8,8)

print(np.linalg.det(ecc_rotation_matrix_opt_T))
print(ecc_rotation_matrix_opt_T @ ecc_rotation_matrix_opt_T.T)

print("original\n", ecc_rotation_matrix_T)
print("optimized\n", ecc_rotation_matrix_opt_T)

Phi = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ sim['x'])

fig, axs = plt.subplots(2,8,figsize=(20, 5))
for i, pl in enumerate(planets):
    axs[0][i].set_title(pl)
    pts = sim['x'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Phi[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
N = sim['x'].shape[0]
X = jnp.concatenate((sim['x'], -1j*np.conj(sim['x'])), axis=0)
J = jnp.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])

def objective(M, J, X):
    M = jnp.reshape(M, (N*2, N*2))
    
    symplectic_loss = ((M.T @ J @ M - J)**2).sum()

    Phi = M.T @ X
    J_loss = ((jnp.abs(Phi) - jnp.abs(Phi).mean(axis=1)[..., None]) ** 2)
    J_loss = J_loss.sum()
    # J_loss = (J_loss / jnp.pow(jnp.concat((masses, masses))[..., None], 1/4)).sum()
    
    loss = symplectic_loss + J_loss * 10
    return loss
obj_and_grad = jax.jit(jax.value_and_grad(lambda M: objective(M, J, X)))

np.random.seed(0)
initial = np.eye(2*N) + np.random.normal(0, 0.1, (N*2, N*2))
# initial = jnp.block([[np.zeros((N, N)), ecc_rotation_matrix_T], [-np.eye(N), np.zeros((N, N))]])
sol = minimize(obj_and_grad, initial.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
ecc_rotation_matrix_opt_T = sol.x.reshape((N*2, N*2))

print(ecc_rotation_matrix_opt_T.T @ J @ ecc_rotation_matrix_opt_T)
Phi = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ X)[:N]

fig, axs = plt.subplots(2,8,figsize=(20, 5))
for i, pl in enumerate(planets):
    axs[0][i].set_title(pl)
    pts = sim['x'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Phi[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
# %config InlineBackend.figure_format = 'retina'
# plt.rcParams['font.size'] = 12
X = jnp.concatenate((full_sim['x'], -1j*np.conj(full_sim['x'])), axis=0)
Phi_full = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ X)[:N]
# fig, axs = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
fig, axs = plt.subplots(2, 2, figsize=(15, 7), sharex=True)
axs[0][0].plot(full_sim['time'] * TO_YEAR, np.abs(full_sim['x'][0]))
axs[1][0].plot(full_sim['time'] * TO_YEAR, np.abs(Phi_full[0]))
axs[0][0].set_ylim(0, 6.5e-5)
axs[1][0].set_ylim(0, 6.5e-5)
axs[0][0].set_title("Mercury Before")
axs[1][0].set_title("Mercury After")
axs[0][0].axvline(full_sim['time'][keep_first] * TO_YEAR, linestyle="--", color="black")
axs[1][0].axvline(full_sim['time'][keep_first] * TO_YEAR, linestyle="--", color="black")
axs[1][0].set_xlabel("Years")

axs[0][1].plot(full_sim['time'] * TO_YEAR, np.abs(full_sim['x'][4]))
axs[1][1].plot(full_sim['time'] * TO_YEAR, np.abs(Phi_full[4]))
axs[0][1].set_ylim(0, 3e-3)
axs[1][1].set_ylim(0, 3e-3)
axs[0][1].set_title("Jupiter Before")
axs[1][1].set_title("Jupiter After")
axs[0][1].axvline(full_sim['time'][keep_first] * TO_YEAR, linestyle="--", color="black")
axs[1][1].axvline(full_sim['time'][keep_first] * TO_YEAR, linestyle="--", color="black")
axs[1][1].set_xlabel("Years")
# plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=0.2)
# %config InlineBackend.figure_format = ''
# plt.rcParams['font.size'] = 10

np.savez_compressed("planet_aa",
                    mercury_before=full_sim['x'][0],
                    mercury_after=Phi_full[0],
                    jupiter_before=full_sim['x'][4],
                    jupiter_after=Phi_full[4],
                    sim_time=full_sim['time'] * TO_YEAR,
                    training_time=full_sim['time'][keep_first] * TO_YEAR)
# %%
def planet_fmft(time, x, N=14, display=False):
    planet_ecc_fmft = {}
    for i,pl in enumerate(planets):
        planet_ecc_fmft[pl] = fmft(time,x[i],N)
        planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))

        if display:
            print("")
            print(pl)
            print("-------")
            for g in planet_e_freqs[:8]:
                print("{:+07.3f} \t {:0.8f}".format(g * TO_ARCSEC_PER_YEAR, np.abs(planet_ecc_fmft[pl][g])))
            # print("s")
            # print("-------")
            # for s in planet_inc_freqs[:4]:
            #     print("{:+07.3f} \t {:0.6f}".format(s * TO_ARCSEC_PER_YEAR, np.abs(planet_inc_fmft[pl][s])))
    
    return planet_ecc_fmft

planet_ecc_fmft = planet_fmft(sim['time'], Phi, display=True)
# %%
possible_k = []
# SECOND ORDER
for a in range(N):
    for b in range(a+1, N):
        if a == b:
            continue
        k = np.zeros(N, dtype=int)
        k[a] += 1
        k[b] -= 1
        possible_k.append(k)

# THIRD ORDER
for a in range(N):
    for b in range(a,N):
        for c in range(N):
            if c==a:
                continue
            if c==b:
                continue
            k = np.zeros(N, dtype=int)
            k[a] +=1
            k[b] +=1
            k[c] -=1
            possible_k.append(k)

possible_k = np.array(possible_k)
# %%
g_vec = np.zeros(8)

g_vec[0] = list(planet_ecc_fmft['Mercury'].keys())[0]
g_vec[1] = list(planet_ecc_fmft['Venus'].keys())[0]
g_vec[2] = list(planet_ecc_fmft['Earth'].keys())[0]
g_vec[3] = list(planet_ecc_fmft['Mars'].keys())[0]
g_vec[4] = list(planet_ecc_fmft['Jupiter'].keys())[0]
g_vec[5] = list(planet_ecc_fmft['Saturn'].keys())[0]
g_vec[6] = list(planet_ecc_fmft['Uranus'].keys())[0]
g_vec[7] = list(planet_ecc_fmft['Neptune'].keys())[0]

g_amp = np.zeros(8, dtype=np.complex128)
g_amp[0] = planet_ecc_fmft['Mercury'][g_vec[0]]
g_amp[1] = planet_ecc_fmft['Venus'][g_vec[1]]
g_amp[2] = planet_ecc_fmft['Earth'][g_vec[2]]
g_amp[3] = planet_ecc_fmft['Mars'][g_vec[3]]
g_amp[4] = planet_ecc_fmft['Jupiter'][g_vec[4]]
g_amp[5] = planet_ecc_fmft['Saturn'][g_vec[5]]
g_amp[6] = planet_ecc_fmft['Uranus'][g_vec[6]]
g_amp[7] = planet_ecc_fmft['Neptune'][g_vec[7]]

print(g_vec * TO_ARCSEC_PER_YEAR)
print(g_amp)
# %%
combs = []
for pl in planets:
    print(pl) 
    print("-"*len(pl))
    print("kvec \t\t\t omega \t err. \t amplitude")
    comb = {}
    for k in possible_k:
        omega = k @ g_vec
        omega_N,amp = closest_key_entry(planet_ecc_fmft[pl],omega)
        omega_error = np.abs(omega_N/omega-1)
        omega_error_pct = np.abs((omega_N - omega)/omega)
        if omega_error<1e-4:# and omega_error_pct < 0.01:
            print (k,"\t{:+07.3f}\t{:.1g},\t{:.1g}".format(omega*TO_ARCSEC_PER_YEAR,omega_error,np.abs(amp)))
            comb[tuple(k)] = amp
    combs.append(comb)
# %%
# for pl in combs:
#     to_del = []
#     for comb in pl.keys():
#         if comb != (-1, 2, 0, 0, 0):
#             to_del.append(comb)
#     for d in to_del:
#         del pl[d]
# %%
x_val = Phi
x = [sympy.Symbol("X_"+str(i)) for i in range(N)]
x_bar_0 = [sympy.Symbol("\\bar X^{(0)}_"+str(i)) for i in range(N)]

x_bars = [x_bar_0]
subs = {x_bar_0[i]: x[i] for i in range(N)}
x_val_subs = {x[i]: x_val[:, i] for i in range(N)}

iterations = 1
for i in range(1, iterations+1):
    x_bar_i = [sympy.Symbol(f"\\bar X^{{({i})}}_"+str(j)) for j in range(N)]
    # loop through each object
    for j in range(N):
        # to first order the coordinate is the original coordinate
        x_bar_i_j = x_bars[-1][j]
        # correct for each combination
        for k,amp in combs[j].items():
            term = amp
            # loop through each object
            for k_idx in range(N):
                # add each object the correct number of times
                for l in range(np.abs(k[k_idx])):
                    term *= x_bars[-1][k_idx]/g_amp[k_idx] if k[k_idx] > 0 else x_bars[-1][k_idx].conjugate()/np.conj(g_amp[k_idx])
            x_bar_i_j -= term
        subs[x_bar_i[j]] = x_bar_i_j
    x_bars.append(x_bar_i)

x_bar_n = x_bars[-1]
for i in range(iterations+1):
    for j in range(N):
        x_bar_n[j] = x_bar_n[j].subs(subs)
x_bar_n
# %%
Phi_second_order = np.array([sympy.lambdify(x, x_bar_n[i], 'numpy')(*x_val) for i in range(N)])
_ = planet_fmft(sim['time'], Phi_second_order, display=True)
# %%
fig, axs = plt.subplots(2,8,figsize=(20, 5))
for i, pl in enumerate(planets):
    axs[0][i].set_title(pl)
    pts = Phi[i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Phi_second_order[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%