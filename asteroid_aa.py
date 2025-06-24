# %%
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jnr
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.optimize import minimize

import tensorflow as tf
from flax import nnx
import optax
import orbax.checkpoint as ocp
from nn.sympnet.la import LA_Layer
from nn.utils import get_pq, get_x

import rebound as rb

from celmech.nbody_simulation_utilities import get_simarchive_integration_results
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft

import sympy

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
dataset_path = Path('datasets') / 'asteroid_integration' / 'outer_planets'
base_desn = '00001'
TO_ARCSEC_PER_YEAR = 60*60*180/np.pi * (2*np.pi)
# %%
sims = list(dataset_path.glob('*.sa'))
desns = list(map(lambda p: p.stem.split('_')[-1], sims))
desns = sorted(desns)
# %%
rb_sim = rb.Simulation(str(dataset_path/'planets.bin'))
masses = np.array(list(map(lambda ps: ps.m, rb_sim.particles))[1:] + [1e-11], dtype=np.float64)

def load_sim(desn):
    results = get_simarchive_integration_results(str(dataset_path / f"{'_'.join(sims[0].stem.split('_')[:-1])}_{desn}.sa"), coordinates='heliocentric')
    
    # results['X'] = np.sqrt(2*(1-np.sqrt(1-results['e']**2))) * np.exp(1j * results['pomega'])
    # results['Y'] = (1-results['e']**2)**(0.25) * np.sin(0.5 * results['inc'] ) * np.exp(1j * results['Omega']) 

    # prefactor = np.sqrt(masses)[..., None].repeat(results['X'].shape[1], axis=-1) * np.power(rb_sim.G * rb_sim.particles[0].m * results['a'], 1/4)
    # results['G'] = prefactor * results['X']
    # results['F'] = prefactor * results['Y']
    m = masses[..., None].repeat(results['a'].shape[1], axis=-1)
    G = 1
    beta = ((1 * m) / (1 + m))
    mu = G * (1 + m)
    results['Lambda'] = beta * np.sqrt(mu * results['a'])
    
    M = results['l'] - results['pomega']
    results['lambda'] = M + results['pomega']

    results['x'] = np.sqrt(results['Lambda']) * np.sqrt(1 - np.sqrt(1-results['e']**2)) * np.exp(1j * results['pomega'])
    results['y'] = np.sqrt(2 * results['Lambda']) * np.power(1-results['e']**2, 1/4) * np.sin(results['inc']/2) * np.exp(1j * results['Omega'])

    # results['x'] = np.float64(results['x'])
    # results['y'] = np.float64(results['y'])

    results['G'] = results['x']
    results['F'] = results['y']

    # coordinate pairs are:
    # - Lambda, Lambda
    # - x, -i * x_bar
    # - y, -i * y_bar

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

    planet_inc_fmft[pl] = fmft(base_sim['time'],base_sim['F'][i],8)
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
freq_thresh = 0.05
ecc_rotation_matrix_T_base = np.zeros((5, 5))

# planet ecc in terms of planet modes
mode_angle = [0, 0, 0, 0, 0]
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
        ecc_rotation_matrix_T_base[i][j] += matrix_entry
    print()

mode_angle = [0, 0, 0, 0, 0]
inc_rotation_matrix_T_base = np.zeros((5, 5))
for i, pl in enumerate(planets):
    planet_i_freqs = np.array(list(planet_inc_fmft[pl].keys()))
    for j, s in enumerate(s_vec):
        found_s = find_nearest(planet_i_freqs, s)
        if np.abs((found_s - s)/s) > freq_thresh:
            continue
        matrix_entry = np.abs(planet_inc_fmft[pl][found_s])
        if mode_angle[i] == 0:
            mode_angle[i] = np.angle(planet_inc_fmft[pl][found_s])
        if mode_angle[i] != 0 and np.abs(mode_angle[i] - np.angle(planet_inc_fmft[pl][found_s])) > np.pi/2:
            matrix_entry *= -1
        inc_rotation_matrix_T_base[i][j] = matrix_entry
# %%
for desn in desns:
    sim = load_sim(desn)
    asteroid_ecc_fmft = fmft(sim['time'],sim['G'][-1],14)
    asteroid_inc_fmft = fmft(sim['time'],sim['F'][-1],8)

    ecc_rotation_matrix_T = ecc_rotation_matrix_T_base.copy()
    inc_rotation_matrix_T = inc_rotation_matrix_T_base.copy()

    asteroid_e_freqs = np.array(list(asteroid_ecc_fmft.keys()))
    asteroid_i_freqs = np.array(list(asteroid_inc_fmft.keys()))
    print("Asteroid\ng\n--------")
    for g in asteroid_e_freqs[:6]:
        print("{:+07.3f} \t {:0.6f}".format(g * TO_ARCSEC_PER_YEAR, np.abs(asteroid_ecc_fmft[g])))
    print('\ns\n--------')
    for s in asteroid_i_freqs[:4]:
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

with np.printoptions(suppress=True, precision=5):
    print("ecc")
    print(ecc_rotation_matrix_T)
    print("inc")
    print(inc_rotation_matrix_T)
# %%
fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i,pl in enumerate(planets):
    axs[0][i].set_title(pl)
    print(pl)
    zsoln = np.zeros_like(sim['time'], dtype=np.complex128)
    cnt = 0
    for freq, amp in planet_ecc_fmft[pl].items():
        found_g = find_nearest(g_vec, freq)
        with np.printoptions(suppress=True, precision=3):
            print(np.array([found_g*TO_ARCSEC_PER_YEAR, freq*TO_ARCSEC_PER_YEAR, (found_g - freq)/freq * 100]), np.abs((found_g - freq)/freq) < 0.01, np.round(np.abs(amp), 5))
        if np.abs(found_g - freq)/found_g < freq_thresh:
            zsoln += amp * np.exp(1j*freq*sim['time'])
            cnt += 1
        if cnt > 1: break
    print()

    pts = sim['G'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = zsoln
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')

    # base_sim['G'][i] = zsoln
    # sim['G'][i] = zsoln
# %%
np.set_printoptions(suppress=True, precision=4)
# %%
ecc_rotation_matrix_T = ecc_rotation_matrix_T / np.linalg.norm(ecc_rotation_matrix_T, axis=0)
print(np.linalg.det(ecc_rotation_matrix_T))

print(ecc_rotation_matrix_T)

Phi = (np.linalg.inv(ecc_rotation_matrix_T) @ sim['G'])
Phi[-1] *= 1e7

fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i, pl in enumerate(planets + ('Asteroid',)):
    axs[0][i].set_title(pl)
    pts = sim['G'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Phi[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
# transformedY = (sim['F'].T @ inc_rotation_matrix_T.T).T
transformedY = (np.linalg.inv(inc_rotation_matrix_T) @ sim['F'])

fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i in range(5):
    pts = sim['F'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = transformedY[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
def objective(R, G, m):
    R = jnp.reshape(R, (5,5))
    # get a valid rotation matrix from W
    # U, S, Vh = np.linalg.svd(W)
    # R = U @ Vh

    rotation_loss = ((jnp.eye(5) - R @ R.T) ** 2).sum() #+ (np.linalg.det(R) - 1) ** 2

    Phi = R.T @ G

    J_loss = ((jnp.abs(Phi) - jnp.abs(Phi).mean(axis=1)[..., None]) ** 2).sum()

    loss = rotation_loss + J_loss
    return loss

obj_and_grad = jax.jit(jax.value_and_grad(lambda R: objective(R, sim['G'], masses)))

sol = minimize(obj_and_grad, ecc_rotation_matrix_T.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
ecc_rotation_matrix_opt_T = sol.x.reshape(5,5)

print(np.linalg.det(ecc_rotation_matrix_opt_T))
print(ecc_rotation_matrix_opt_T @ ecc_rotation_matrix_opt_T.T)

print("original\n", ecc_rotation_matrix_T)
print("optimized\n", ecc_rotation_matrix_opt_T)

Phi = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ sim['G'])

fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i, pl in enumerate(planets + ('Asteroid',)):
    axs[0][i].set_title(pl)
    pts = sim['G'][i]
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
    # J_loss = J_loss.sum()
    J_loss = (J_loss / jnp.pow(jnp.concat((masses, masses))[..., None], 1/4)).sum()
    
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

fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i, pl in enumerate(planets + ('Asteroid',)):
    axs[0][i].set_title(pl)
    pts = sim['G'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Phi[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
fig, axs = plt.subplots(2, 1)
axs[0].plot(np.real(Phi[-1, :1000]))
axs[1].plot(np.real(Phi[-1, :100]))
# %%
def planet_fmft(time, x, N=14, display=False):
    planet_ecc_fmft = {}
    for i,pl in enumerate(planets + ("Asteroid",)):
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

planet_ecc_fmft = planet_fmft(base_sim['time'], Phi, display=True)
# %%
possible_k = []
# SECOND ORDER
# for a in range(N):
#     for b in range(a+1, N):
#         if a == b:
#             continue
#         k = np.zeros(N, dtype=int)
#         k[a] += 1
#         k[b] -= 1
#         possible_k.append(k)

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
g_vec = np.zeros(5)

g_vec[0] = list(planet_ecc_fmft['Jupiter'].keys())[0]
g_vec[1] = list(planet_ecc_fmft['Saturn'].keys())[0]
g_vec[2] = list(planet_ecc_fmft['Uranus'].keys())[0]
g_vec[3] = list(planet_ecc_fmft['Neptune'].keys())[0]
g_vec[4] = list(planet_ecc_fmft['Asteroid'].keys())[0]

print(g_vec * TO_ARCSEC_PER_YEAR)
# %%
combs = []
for pl in planets + ("Asteroid",):
    print(pl) 
    print("-"*len(pl))
    print("kvec \t\t\t omega \t err. \t amplitude")
    comb = {}
    for k in possible_k:
        omega = k @ g_vec
        omega_N,amp = closest_key_entry(planet_ecc_fmft[pl],omega)
        omega_error = np.abs(omega_N/omega-1)
        omega_error_pct = np.abs((omega_N - omega)/omega)
        if omega_error<0.001:# and omega_error_pct < 0.01:
            print (k,"\t{:+07.3f}\t{:.1g},\t{:.1g}".format(omega*TO_ARCSEC_PER_YEAR,omega_error,np.abs(amp)))
            comb[tuple(k)] = amp
    combs.append(comb)
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
    for j in range(N):
        x_bar_i_j = x_bars[-1][j]
        for k,amp in combs[j].items():
            term = amp
            for k_idx in range(N):
                if k[k_idx] == 0:
                    continue
                term *= x_bars[-1][k_idx] if k[k_idx] > 0 else x_bars[-1][k_idx].conjugate()
            x_bar_i_j += term
        subs[x_bar_i[j]] = x_bar_i_j
    x_bars.append(x_bar_i)

x_bar_n = x_bars[-1]
for i in range(iterations+1):
    for j in range(N):
        x_bar_n[j] = x_bar_n[j].subs(subs)
x_bar_n
# %%
Phi_second_order = np.array([sympy.lambdify(x, x_bar_n[i], 'numpy')(*x_val) for i in range(N)])
_ = planet_fmft(base_sim['time'], Phi_second_order, display=True)
# %%
fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i, pl in enumerate(planets + ('Asteroid',)):
    axs[0][i].set_title(pl)
    pts = Phi[i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Phi_second_order[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
# sol = minimize(objective, inc_rotation_matrix_T.reshape(-1), args=(sim['F'], masses), options={'gtol': 1e-8, 'disp': True})
# inc_rotation_matrix_opt_T = sol.x.reshape(5,5)

# with np.printoptions(suppress=True, precision=4):
#     print(np.linalg.det(inc_rotation_matrix_opt_T))
#     print(inc_rotation_matrix_opt_T @ inc_rotation_matrix_opt_T.T)

# with np.printoptions(suppress=True, precision=4):
#     print("original\n", inc_rotation_matrix_T)
#     print("optimized\n", inc_rotation_matrix_opt_T)

# transformedY = (np.linalg.inv(inc_rotation_matrix_opt_T) @ sim['F'])

# fig, axs = plt.subplots(2,5,figsize=(15, 5))
# for i, pl in enumerate(planets + ('Asteroid',)):
#     axs[0][i].set_title(pl)
#     pts = sim['F'][i]
#     axs[0][i].plot(np.real(pts), np.imag(pts))
#     axs[0][i].set_aspect('equal')
#     pts = transformedY[i]
#     axs[1][i].plot(np.real(pts), np.imag(pts))
#     axs[1][i].set_aspect('equal')
# # %%
# fig, axs = plt.subplots(4, 1)
# asteroid_phi = Phi[-1][:100]
# axs[0].plot(np.angle(asteroid_phi), np.abs(asteroid_phi))
# axs[1].plot(np.abs(asteroid_phi))
# axs[2].plot(np.angle(asteroid_phi))
# axs[3].plot(np.gradient(np.angle(asteroid_phi)))
# # %%
# angle = np.angle(asteroid_phi)
# plt.plot(sim['time'][:100], jnp.gradient(jnp.sin(angle)))
# plt.plot(sim['time'][:100], jnp.cos(angle) * (1/(1.5 * jnp.pi)))
# %%