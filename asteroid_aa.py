# %%
from pathlib import Path
import copy
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

def symmetrize_axes(axes):
    y_max = np.max(np.abs(axes.get_ylim()))
    x_max = np.max(np.abs(axes.get_xlim()))

    ax_max = np.max([x_max, y_max])

    axes.set_ylim(ymin=-ax_max, ymax=ax_max)
    axes.set_xlim(xmin=-ax_max, xmax=ax_max)
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
N = len(masses)

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
    planet_ecc_fmft[pl] = fmft(base_sim['time'],base_sim['x'][i],14)
    planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))
    planet_e_freqs_arcsec_per_yr = planet_e_freqs * TO_ARCSEC_PER_YEAR

    planet_inc_fmft[pl] = fmft(base_sim['time'],base_sim['y'][i],8)
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
    asteroid_ecc_fmft = fmft(sim['time'],sim['x'][-1],14)
    asteroid_inc_fmft = fmft(sim['time'],sim['y'][-1],8)

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

    pts = sim['x'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = zsoln
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')

    # base_sim['G'][i] = zsoln
    # sim['G'][i] = zsoln
# %%
np.set_printoptions(suppress=True, precision=4, linewidth=100)
# %%
ecc_rotation_matrix_T = ecc_rotation_matrix_T / np.linalg.norm(ecc_rotation_matrix_T, axis=0)
print(np.linalg.det(ecc_rotation_matrix_T))

print(ecc_rotation_matrix_T)

Phi = (np.linalg.inv(ecc_rotation_matrix_T) @ sim['x'])
Phi[-1] *= 1e7

fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i, pl in enumerate(planets + ('Asteroid',)):
    axs[0][i].set_title(pl)
    pts = sim['x'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Phi[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
# Theta = (sim['F'].T @ inc_rotation_matrix_T.T).T
Theta = (np.linalg.inv(inc_rotation_matrix_T) @ sim['y'])

fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i in range(5):
    pts = sim['y'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Theta[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
def objective(R, G, m):
    R = jnp.reshape(R, (5,5))
    # get a valid rotation matrix from W
    # U, S, Vh = np.linalg.svd(W)
    # R = U @ Vh

    rotation_loss = ((jnp.eye(5) - R @ R.T) ** 2).sum()# + (jnp.linalg.det(R) - 1) ** 2

    Phi = R.T @ G

    J_approx = jnp.abs(Phi).mean(axis=1)
    J_loss = ((jnp.abs(Phi) - J_approx[..., None]) ** 2).sum()
    
    # norm_phi = jnp.real(Phi/J_approx[..., None])
    # orthog_loss = (((norm_phi @ norm_phi.T) * (jnp.ones((5,5))-jnp.eye(5))) ** 2).sum()

    off_diag_weight = 1 / jnp.pow(jnp.outer(J_approx, J_approx), 1/4)
    off_diag_loss = (((jnp.ones((5,5))-jnp.eye(5)) * R.T * off_diag_weight) ** 2).sum()

    loss = rotation_loss + J_loss + off_diag_loss * 1e-10 #+ orthog_loss * 1e-6
    # loss = rotation_loss + J_loss + off_diag_loss * 1e-10 + orthog_loss * 1e-6
    return loss

obj_and_grad = jax.jit(jax.value_and_grad(lambda R: objective(R, sim['x'], masses)))

sol = minimize(obj_and_grad, ecc_rotation_matrix_T.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
ecc_rotation_matrix_opt_T = sol.x.reshape(5,5)

print(np.linalg.det(ecc_rotation_matrix_opt_T))
print(ecc_rotation_matrix_opt_T @ ecc_rotation_matrix_opt_T.T)

print("original\n", ecc_rotation_matrix_T)
print("optimized\n", ecc_rotation_matrix_opt_T)

Phi = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ sim['x'])

fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i, pl in enumerate(planets + ('Asteroid',)):
    axs[0][i].set_title(pl)
    pts = sim['x'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Phi[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
# N = sim['x'].shape[0]
# X = jnp.concatenate((sim['x'], -1j*np.conj(sim['x'])), axis=0)
# J = jnp.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])

# def objective(M, J, X):
#     M = jnp.reshape(M, (N*2, N*2))
    
#     symplectic_loss = ((M.T @ J @ M - J)**2).sum()

#     Phi = M.T @ X
#     J_loss = ((jnp.abs(Phi) - jnp.abs(Phi).mean(axis=1)[..., None]) ** 2) #/ jnp.abs(Phi).mean(axis=1)[..., None]
#     J_loss = J_loss.sum()
#     # J_loss = J_loss[-2].sum()
    
#     off_diag_loss = (((jnp.ones((N*2,N*2))-jnp.eye(N*2)) * M) ** 2).sum()
    
#     loss = symplectic_loss + J_loss*10 + off_diag_loss*0.05
#     return loss
# obj_and_grad = jax.jit(jax.value_and_grad(lambda M: objective(M, J, X)))

# np.random.seed(0)
# # initial = np.eye(2*N) + np.random.normal(0, 0.1, (N*2, N*2))
# initial = jnp.block([[ecc_rotation_matrix_T, np.zeros((N, N))], [np.zeros((N, N)), ecc_rotation_matrix_T]])
# initial += np.random.normal(0, 0.05, (N*2, N*2))
# sol = minimize(obj_and_grad, initial.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
# ecc_rotation_matrix_opt_T = sol.x.reshape((N*2, N*2))

# # print(ecc_rotation_matrix_opt_T.T @ J @ ecc_rotation_matrix_opt_T)
# print(ecc_rotation_matrix_opt_T)
# Phi = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ X)[:N]

# fig, axs = plt.subplots(2,5,figsize=(15, 5))
# for i, pl in enumerate(planets + ('Asteroid',)):
#     axs[0][i].set_title(pl)
#     pts = sim['x'][i]
#     axs[0][i].plot(np.real(pts), np.imag(pts))
#     axs[0][i].set_aspect('equal')
#     pts = Phi[i]
#     axs[1][i].plot(np.real(pts), np.imag(pts))
#     axs[1][i].set_aspect('equal')
# %%
def objective(R, G, m):
    R = jnp.reshape(R, (5,5))

    rotation_loss = ((jnp.eye(5) - R @ R.T) ** 2).sum()# + (jnp.linalg.det(R) - 1) ** 2

    Phi = R.T @ G

    J_approx = jnp.abs(Phi).mean(axis=1)
    J_loss = ((jnp.abs(Phi) - J_approx[..., None]) ** 2).sum()

    off_diag_weight = 1 / jnp.pow(jnp.outer(J_approx, J_approx), 1/4)
    off_diag_loss = (((jnp.ones((5,5))-jnp.eye(5)) * R.T * off_diag_weight) ** 2).sum()

    loss = rotation_loss + J_loss + off_diag_loss * 1e-10
    return loss

obj_and_grad = jax.jit(jax.value_and_grad(lambda R: objective(R, sim['y'], masses)))

sol = minimize(obj_and_grad, ecc_rotation_matrix_T.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
inc_rotation_matrix_opt_T = sol.x.reshape(5,5)

print(np.linalg.det(inc_rotation_matrix_opt_T))
print(inc_rotation_matrix_opt_T @ inc_rotation_matrix_opt_T.T)

Theta = (np.linalg.inv(inc_rotation_matrix_opt_T) @ sim['y'])

fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i, pl in enumerate(planets + ('Asteroid',)):
    axs[0][i].set_title(pl)
    pts = sim['y'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Theta[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
# N = sim['y'].shape[0]
# Y = jnp.concatenate((sim['y'], -1j*np.conj(sim['y'])), axis=0)

# def objective(M, J, Y):
#     M = jnp.reshape(M, (N*2, N*2))
    
#     symplectic_loss = ((M.T @ J @ M - J)**2).sum()

#     Phi = M.T @ Y
#     J_loss = ((jnp.abs(Phi) - jnp.abs(Phi).mean(axis=1)[..., None]) ** 2)
#     J_loss = J_loss.sum()
#     # J_loss = (J_loss / jnp.pow(jnp.concat((masses, masses))[..., None], 1/4)).sum()
    
#     loss = symplectic_loss + J_loss * 10
#     return loss
# obj_and_grad = jax.jit(jax.value_and_grad(lambda M: objective(M, J, Y)))

# np.random.seed(1)
# # initial = np.eye(2*N) + np.random.normal(0, 0.1, (N*2, N*2))
# initial = jnp.block([[inc_rotation_matrix_opt_T, np.zeros((N,N))], [np.zeros((N, N)), inc_rotation_matrix_opt_T]])
# sol = minimize(obj_and_grad, initial.reshape(-1), jac=True, options={'gtol': 1e-8, 'disp': True})
# inc_rotation_matrix_opt_T = sol.x.reshape((N*2, N*2))

# print(inc_rotation_matrix_opt_T.T @ J @ inc_rotation_matrix_opt_T)
# Theta = (np.linalg.inv(initial) @ Y)[:N]

# fig, axs = plt.subplots(2,5,figsize=(15, 5))
# for i, pl in enumerate(planets + ('Asteroid',)):
#     axs[0][i].set_title(pl)
#     pts = sim['y'][i]
#     axs[0][i].plot(np.real(pts), np.imag(pts))
#     axs[0][i].set_aspect('equal')
#     pts = Theta[i]
#     axs[1][i].plot(np.real(pts), np.imag(pts))
#     axs[1][i].set_aspect('equal')
# %%
planet_ecc_fmft = {}
planet_inc_fmft = {}
for i,pl in enumerate(planets + ("Asteroid",)):
    planet_ecc_fmft[pl] = fmft(base_sim['time'],Phi[i],14)
    planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))

    planet_inc_fmft[pl] = fmft(base_sim['time'],Theta[i],14)
    planet_i_freqs = np.array(list(planet_inc_fmft[pl].keys()))

    print("")
    print(pl)
    print("-------")
    for g in planet_e_freqs[:8]:
        print(f"{g * TO_ARCSEC_PER_YEAR:+07.3f} \t {np.abs(planet_ecc_fmft[pl][g]):0.8f} ∠{np.angle(planet_ecc_fmft[pl][g]):.2f}")
    print("s")
    print("-------")
    for s in planet_i_freqs[:4]:
        print(f"{s * TO_ARCSEC_PER_YEAR:+07.3f} \t {np.abs(planet_inc_fmft[pl][s]):0.6f} ∠{np.angle(planet_inc_fmft[pl][s]):.2f}")
    
    # return planet_ecc_fmft, planet_inc_fmft

# planet_ecc_fmft, planet_inc_fmft = planet_fmft(base_sim['time'], Phi, Theta, display=True)
# %%
g_vec = np.zeros(5)
s_vec = np.zeros(5)

g_vec[0] = list(planet_ecc_fmft['Jupiter'].keys())[0]
g_vec[1] = list(planet_ecc_fmft['Saturn'].keys())[0]
g_vec[2] = list(planet_ecc_fmft['Uranus'].keys())[0]
g_vec[3] = list(planet_ecc_fmft['Neptune'].keys())[0]
g_vec[4] = list(planet_ecc_fmft['Asteroid'].keys())[0]

s_vec[0] = list(planet_inc_fmft['Jupiter'].keys())[0]
s_vec[1] = list(planet_inc_fmft['Saturn'].keys())[0]
s_vec[2] = list(planet_inc_fmft['Uranus'].keys())[0]
s_vec[3] = list(planet_inc_fmft['Neptune'].keys())[0]
s_vec[4] = list(planet_inc_fmft['Asteroid'].keys())[0]

g_amp = np.zeros(5, dtype=np.complex128)
s_amp = np.zeros(5, dtype=np.complex128)

g_amp[0] = planet_ecc_fmft['Jupiter'][g_vec[0]]
g_amp[1] = planet_ecc_fmft['Saturn'][g_vec[1]]
g_amp[2] = planet_ecc_fmft['Uranus'][g_vec[2]]
g_amp[3] = planet_ecc_fmft['Neptune'][g_vec[3]]
g_amp[4] = planet_ecc_fmft['Asteroid'][g_vec[4]]

s_amp[0] = planet_inc_fmft['Jupiter'][s_vec[0]]
s_amp[1] = planet_inc_fmft['Saturn'][s_vec[1]]
s_amp[2] = planet_inc_fmft['Uranus'][s_vec[2]]
s_amp[3] = planet_inc_fmft['Neptune'][s_vec[3]]
s_amp[4] = planet_inc_fmft['Asteroid'][s_vec[4]]

omega_vec = np.concat([g_vec, s_vec])
omega_amp = np.concat([g_amp, s_amp])

s_conserved_idx = np.argmin(np.abs(omega_vec[N:])) + N

print(omega_vec * TO_ARCSEC_PER_YEAR)
print(omega_amp)
# %%
base_planet_list = planets + ("Asteroid",)
psi_planet_list = (tuple(map(lambda pl: pl+"_X", base_planet_list)) + tuple(map(lambda pl: pl+"_Y", base_planet_list)))
Psi = np.concat([Phi, Theta], axis=0)
def get_planet_fmft(pl_list, time, X, N=14, display=False, compareto=None):
    planet_fmft = {}
    for i,pl in enumerate(pl_list):
        fmft_res = fmft(time, X[i], N)
        planet_fmft[pl] = fmft_res
        planet_freqs = np.array(list(planet_fmft[pl].keys()))
        
        if display:
            print("")
            print(pl)
            print("-------")
            for i,f in enumerate(planet_freqs):
                print(f"{f * TO_ARCSEC_PER_YEAR:+07.3f} \t {np.abs(planet_fmft[pl][f]):0.8f}  ∢{np.angle(planet_fmft[pl][f]):.2f}", end='')
                if compareto:
                    ctf = list(compareto[pl].keys())[i]
                    print(f"\t\t{ctf * TO_ARCSEC_PER_YEAR:+07.3f} \t {np.abs(compareto[pl][ctf]):0.8f}  ∢{np.angle(compareto[pl][ctf]):.2f}", end='')
                print()
    return planet_fmft
planet_fmft = get_planet_fmft(psi_planet_list, base_sim['time'], Psi, N=14, display=True)
# %%
def get_k_vecs(order, pl_idx, s_conserved_idx, N, include_negative=False):
    assert order % 2 == 1, "Order must be odd"
    possible_k = []

    # FIRST ORDER
    if order == 1:
        for a in range(N*2):
            if a == pl_idx:
                continue
            k = np.zeros(N*2, dtype=int)
            k[a] = 1
            possible_k.append(k)
    
    # THIRD ORDER
    if order == 3:
        for a in range(N*2):
            for b in range(a,N*2):
                for c in range(N*2):
                    if c==a:
                        continue
                    if c==b:
                        continue
                    k = np.zeros(N*2, dtype=int)
                    k[a] +=1
                    k[b] +=1
                    k[c] -=1
                    if k[s_conserved_idx] != 0:
                        continue
                    possible_k.append(k)
    
    # FIFTH ORDER
    if order == 5:
        for a in range(N*2):
            for b in range(a,N*2):
                for c in range(b, N*2):
                    for d in range(N*2):
                        for e in range(d,N*2):
                            if d==a or d==b or d==c:
                                continue
                            if e==a or e==b or e==c:
                                continue
                            k = np.zeros(N*2, dtype=int)
                            k[a] +=1
                            k[b] +=1
                            k[c] +=1
                            k[d] -=1
                            k[e] -=1
                            if k[s_conserved_idx] != 0:
                                continue
                            possible_k.append(k)
    
    # SEVENTH ORDER
    if order == 7:
        for a in range(N*2):
            for b in range(a,N*2):
                for c in range(b, N*2):
                    for d in range(c, N*2):
                        for e in range(N*2):
                            for f in range(e,N*2):
                                for g in range(f,N*2):
                                    if f==a or f==b or f==c or f==d:
                                        continue
                                    if g==a or g==b or g==c or g==d:
                                        continue
                                    k = np.zeros(N*2, dtype=int)
                                    k[a] +=1
                                    k[b] +=1
                                    k[c] +=1
                                    k[d] +=1
                                    k[e] -=1
                                    k[f] -=1
                                    k[g] -=1
                                    if k[s_conserved_idx] != 0:
                                        continue
                                    possible_k.append(k)

    possible_k = np.array(possible_k)
    if include_negative:
        possible_k = np.concat((possible_k, -possible_k), axis=0)
    return possible_k

def get_combs(order, pl_fmft, pl_list, omega_vec, display=False, include_negative=False, omega_pct_thresh=1e-4, omega_abs_thresh=1e-3):
    combs = []
    for i,pl in enumerate(pl_list):
        if display:
            print()
            print(f"{pl} \t base amp: {np.abs(list(pl_fmft[pl].items())[0][1]):.2g}") 
            print("-"*len(pl))
            print("kvec \t\t\t\t\t omega \t err. \t amplitude")
        comb = {}
        for k in get_k_vecs(order, i, s_conserved_idx, N, include_negative=include_negative):
            omega = k @ omega_vec
            # print(omega*TO_ARCSEC_PER_YEAR, k)
            omega_N,amp = closest_key_entry(pl_fmft[pl],omega)
            omega_pct_error = np.abs(omega_N/omega-1)
            omega_abs_error = np.abs(omega_N - omega)
            if omega_pct_error<omega_pct_thresh and omega_abs_error < omega_abs_thresh:
                print (k,"\t{:+07.3f}\t{:.1g},\t{:.1g}".format(omega*TO_ARCSEC_PER_YEAR,omega_pct_error,np.abs(amp)))
                comb[tuple(k)] = (amp, omega_N)
        combs.append(comb)
    return combs

# def eval_transform(x, x_bars, subs, x_val, num_iter):
#     x_bar_n = x_bars[-1]
#     for _ in range(num_iter):
#         x_bar_n = [x_bar.subs(subs) for x_bar in x_bar_n]
#     x_trans = np.array([sympy.lambdify(x, x_bar_n[i], 'numpy')(*x_val) for i in range(N*2)])
#     return x_trans, x_bar_n

def eval_transform(x_bars, subs):
    x_i_lambda = [sympy.lambdify(x_bars[-2], x_bars[-1][i].subs(subs)) for i in range(N*2)]

    trans = lambda x: np.array([x_lambda(*x) for x_lambda in x_i_lambda])
    return trans

def apply_sequential_transforms(x, transforms):
    for transform in reversed(transforms):
        x = transform(x)
    return x
# %%
x_val = Psi
x = [sympy.Symbol("X_"+str(i)) for i in range(N*2)]
x_bar_0 = [sympy.Symbol("\\bar X^{(0)}_"+str(i)) for i in range(N*2)]

x_bars = [x_bar_0]
subs = {x_bar_0[i]: x[i] for i in range(N*2)}

trans_fns = []

# iterations = [1]
# iterations = [3]
# iterations = [5]
# iterations = [7]
# iterations = [1,3]
# iterations = [1,3,5]
iterations = [1,3,5,7]
for i,order in enumerate(iterations):
    print("#"*10, f"ITERATION {i+1} - ORDER {order}", "#"*10)
    last_x_val = apply_sequential_transforms(x_val, trans_fns)
    last_fmft = get_planet_fmft(psi_planet_list, base_sim['time'], last_x_val, 14, display=False)
    combs = get_combs(order, last_fmft, psi_planet_list, omega_vec, display=True, include_negative=False, omega_pct_thresh=5e-5)

    x_bar_i = [sympy.Symbol(f"\\bar X^{{({i+1})}}_"+str(j)) for j in range(N*2)]

    # loop through each object
    for j in range(N*2):
        # print(psi_planet_list[j])
        # to first order the coordinate is the original coordinate
        x_bar_i_j = x_bars[-1][j]
        # correct for each combination
        for k,(amp, omega) in combs[j].items():
            term = amp
            # term = 1
            # loop through each object
            delta = 0
            for k_idx in range(N*2):
                # add each object the correct number of times
                for l in range(np.abs(k[k_idx])):
                    term *= x_bars[-1][k_idx]/omega_amp[k_idx] if k[k_idx] > 0 else x_bars[-1][k_idx].conjugate()/np.conj(omega_amp[k_idx])
                    # term *= x_bars[-1][k_idx] if k[k_idx] > 0 else x_bars[-1][k_idx].conjugate()
                    # delta += np.angle(omega_amp[k_idx]) if k[k_idx] > 0 else np.angle(np.conj(omega_amp[k_idx]))
                    # print(delta)
            # if omega < 0:
            #     term *= -1
            # print(np.array(k), omega*TO_ARCSEC_PER_YEAR)
            # print(f"{np.angle(amp):.3f}, {delta:.3f}, {np.angle(amp) - delta:.3f}")
            # print(term)
            x_bar_i_j -= term
        subs[x_bar_i[j]] = x_bar_i_j
    x_bars.append(x_bar_i)
    trans_fns.append(eval_transform(x_bars, subs))

Psi_trans = apply_sequential_transforms(x_val, trans_fns)
# %%
import scipy.signal
b, a = scipy.signal.butter(10, 100, 'low', fs=(TO_ARCSEC_PER_YEAR / np.gradient(sim['time']).mean()) * 2 * np.pi) # type: ignore[reportUnknownVariableType]
Psi_filt = scipy.signal.lfilter(b, a, Psi_trans)
# Psi_filt = Psi_trans
# plt.plot(np.real(Psi_filt[3]), np.imag(Psi_filt[3]))
# %%
new_planet_fmft = get_planet_fmft(psi_planet_list, base_sim['time'], Psi_trans, N=14, display=True, compareto=last_fmft)
# %%
fig, axs = plt.subplots(2,2*N,figsize=(30, 5))
for i, pl in enumerate(psi_planet_list):
    axs[0][i].set_title(pl)
    pts = Psi[i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    # symmetrize_axes(axs[0][i])
    pts = Psi_filt[i]
    axs[1][i].plot(np.real(pts)[100:], np.imag(pts)[100:])
    axs[1][i].set_aspect('equal')
    # symmetrize_axes(axs[1][i])
# %%
fig, axs = plt.subplots(2, N,figsize=(15,5))
for i, pl in enumerate(planets + ("Asteroid",)):
    axs[0][i].set_title(pl)
    axs[0][i].plot(sim['time'][100:], Psi_filt[i][100:] * np.conj(Psi_filt[i])[100:])
    axs[1][i].plot(sim['time'][100:], Psi_filt[i+N][100:] * np.conj(Psi_filt[i+N])[100:])
    # axs[0][i].set_ylim(bottom=0)
    # axs[1][i].set_ylim(bottom=0)
axs[0][0].set_ylabel("Eccentricity")
axs[1][0].set_ylabel("Inclination")
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