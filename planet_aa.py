# %%
from pathlib import Path
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.random as jnr
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
import scipy.signal

import rebound as rb

from celmech.nbody_simulation_utilities import get_simarchive_integration_results
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
from celmech.secular import LaplaceLagrangeSystem

import sympy

np.set_printoptions(suppress=True, precision=4, linewidth=100)
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
# dataset_path = Path('datasets') / 'kat_planet_integration'
dataset_path = Path('datasets') / 'planet_integration'
TO_ARCSEC_PER_YEAR = 60*60*180/np.pi * (2*np.pi)
TO_YEAR = 1/(2*np.pi)
N = 8
# %%
rb_sim = rb.Simulation(str(dataset_path/'planets.bin'))
masses = np.array(list(map(lambda ps: ps.m, rb_sim.particles))[1:], dtype=np.float64)

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
# full_sim = load_sim(dataset_path / "planet_integration.236355012349.500000.sa")
# print(full_sim['time'].shape, full_sim['time'][-1] * TO_YEAR)
# for key, val in full_sim.items():
#     full_sim[key] = val[..., :24081]
# print(full_sim['time'].shape, full_sim['time'][-1] * TO_YEAR)
# %%
# train_sim = load_sim(dataset_path / "planet_integration.59088753087.500000.sa")
# train_sim = load_sim(dataset_path / "planet_integration.628318530.50000.sa", filter_freq=None)
train_sim = load_sim(dataset_path / "planet_integration.628318530.40000.sa", filter_freq=200)
# train_sim = load_sim(dataset_path / "planet_integration.sa")
print(train_sim['time'].shape, train_sim['time'][-1] * TO_YEAR)
# keep_first = int(train_sim['time'].shape[0]*0.9)
# keep_first = int(50e3)
# keep_first = train_sim['time'].shape[0] - 1
# print(train_sim['time'][keep_first] * TO_YEAR)
# sim = {}
# for key, val in train_sim.items():
#     sim[key] = val[..., :keep_first]
# train_sim = sim
sim = train_sim

fs_arcsec_per_yr = (TO_ARCSEC_PER_YEAR / np.gradient(train_sim['time']).mean()) * 2 * np.pi
print("sample rate (\"/yr):", fs_arcsec_per_yr)
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
g_vec[2] = list(planet_ecc_fmft['Earth'].keys())[2]
g_vec[3] = list(planet_ecc_fmft['Mars'].keys())[1]
g_vec[4] = list(planet_ecc_fmft['Jupiter'].keys())[0]
g_vec[5] = list(planet_ecc_fmft['Saturn'].keys())[0]
g_vec[6] = list(planet_ecc_fmft['Uranus'].keys())[1]
g_vec[7] = list(planet_ecc_fmft['Neptune'].keys())[0]

s_vec[0] = list(planet_inc_fmft['Mercury'].keys())[0]
s_vec[1] = list(planet_inc_fmft['Venus'].keys())[0]
s_vec[2] = list(planet_inc_fmft['Earth'].keys())[1]
s_vec[3] = list(planet_inc_fmft['Mars'].keys())[0]
s_vec[4] = list(planet_inc_fmft['Jupiter'].keys())[0]
s_vec[5] = list(planet_inc_fmft['Saturn'].keys())[1]
s_vec[6] = list(planet_inc_fmft['Uranus'].keys())[1]
s_vec[7] = list(planet_inc_fmft['Neptune'].keys())[1]


omega_vec = np.concatenate((g_vec,s_vec))
g_and_s_arc_sec_per_yr = omega_vec * TO_ARCSEC_PER_YEAR
print(np.array([5.535, 7.437, 17.357, 17.905, 4.257, 28.77, 3.088, 0.671] + [-5.624, -7.082, -18.837, -17.749, 0.0, -26.348, -2.993, -0.692]))
print(g_and_s_arc_sec_per_yr.round(3))
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

mode_angle = np.zeros(8)
inc_rotation_matrix_T = np.zeros((8,8))
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
        inc_rotation_matrix_T[i][j] = matrix_entry

ecc_rotation_matrix_T = ecc_rotation_matrix_T / np.linalg.norm(ecc_rotation_matrix_T, axis=0)
inc_rotation_matrix_T = inc_rotation_matrix_T / np.linalg.norm(inc_rotation_matrix_T, axis=0)

print("ecc")
print(ecc_rotation_matrix_T)
print("inc")
print(inc_rotation_matrix_T)
# %%
lsys = LaplaceLagrangeSystem.from_Simulation(rb_sim)
ecc_rotation_matrix_T, _ = lsys.diagonalize_eccentricity()
inc_rotation_matrix_T, _ = lsys.diagonalize_inclination()

print("ecc")
print(ecc_rotation_matrix_T)
print("inc")
print(inc_rotation_matrix_T)
# %%
Phi = (np.linalg.inv(ecc_rotation_matrix_T) @ sim['x'])

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
Theta = (np.linalg.inv(inc_rotation_matrix_T) @ sim['y'])

fig, axs = plt.subplots(2,8,figsize=(20, 5))
for i, pl in enumerate(planets):
    axs[0][i].set_title(pl)
    pts = sim['y'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Theta[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
def objective(R, G, m):
    R = jnp.reshape(R, (N,N))

    rotation_loss = ((jnp.eye(N) - R @ R.T) ** 2).sum() + (jnp.linalg.det(R) - 1) ** 2

    Phi = R.T @ G

    J_approx = jnp.abs(Phi).mean(axis=1)
    J_loss = ((jnp.abs(Phi) - J_approx[..., None]) ** 2).sum() / jnp.pow(m, 1/2)[..., None] #/ jnp.pow(J_approx, 1/2)[..., None]
    J_loss = J_loss.sum()

    # off_diag_weight = 1 / jnp.pow(jnp.outer(J_approx, J_approx), 1/2)
    off_diag_weight = jnp.pow(jnp.outer(m, m), -1/4)
    off_diag_weight /= off_diag_weight.max()
    off_diag_weight = jnp.fill_diagonal(off_diag_weight, off_diag_weight.max(), inplace=False)
    # off_diag_loss = (((jnp.ones((N,N))-jnp.eye(N)) * R.T * off_diag_weight) ** 2).sum()
    off_diag_loss = (((R.T - jnp.eye(N)) ** 2) * off_diag_weight).sum()

    # on_diag_loss = (jnp.diag(R) - 1) ** 2
    # on_diag_loss = on_diag_loss.sum()

    loss = rotation_loss*1e-1 + J_loss*1e-2 + off_diag_loss * 1e-3 #+ on_diag_loss * 1e-1
    return loss

obj_and_grad = jax.jit(jax.value_and_grad(lambda R: objective(R, sim['x'], masses)))

sol = minimize(obj_and_grad, ecc_rotation_matrix_T.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
ecc_rotation_matrix_opt_T = sol.x.reshape(N,N)

print(np.linalg.det(ecc_rotation_matrix_opt_T))
print(ecc_rotation_matrix_opt_T @ ecc_rotation_matrix_opt_T.T)

print("original\n", ecc_rotation_matrix_T)
print("optimized\n", ecc_rotation_matrix_opt_T)

Phi = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ sim['x'])

fig, axs = plt.subplots(2,8,figsize=(20, 5))
plt.suptitle("Eccentricity", fontsize=14)
for i, pl in enumerate(planets):
    axs[0][i].set_title(pl)
    pts = sim['x'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Phi[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
# def objective(R, G, m):
#     R = jnp.reshape(R, (N,N))

#     rotation_loss = ((jnp.eye(N) - R @ R.T) ** 2).sum() + (jnp.linalg.det(R) - 1) ** 2

#     Theta = R.T @ G

#     J_approx = jnp.abs(Theta).mean(axis=1)
#     J_loss = ((jnp.abs(Theta) - J_approx[..., None]) ** 2).sum()

#     off_diag_weight = 1 / jnp.pow(jnp.outer(J_approx, J_approx), 1/2)
#     off_diag_weight /= off_diag_weight.max()
#     off_diag_weight = jnp.fill_diagonal(off_diag_weight, off_diag_weight.max()*5, inplace=False)
#     off_diag_loss = (((jnp.ones((N,N))-jnp.eye(N)) * R.T * off_diag_weight) ** 2).sum()
#     # off_diag_loss = (((R.T - jnp.eye(N)) ** 2) * off_diag_weight).sum()

#     # on_diag_loss = (jnp.diag(R) - 1) ** 2
#     # on_diag_loss = on_diag_loss.sum()

#     loss = rotation_loss*1e1 + J_loss*1e1 + off_diag_loss * 1e-1 #+ on_diag_loss * 1e-1
#     return loss

obj_and_grad = jax.jit(jax.value_and_grad(lambda R: objective(R, sim['y'], masses)))

sol = minimize(obj_and_grad, ecc_rotation_matrix_T.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
inc_rotation_matrix_opt_T = sol.x.reshape(N,N)

print(np.linalg.det(inc_rotation_matrix_opt_T))
print(inc_rotation_matrix_opt_T @ inc_rotation_matrix_opt_T.T)

print("original\n", inc_rotation_matrix_T)
print("optimized\n", inc_rotation_matrix_opt_T)

Theta = (np.linalg.inv(inc_rotation_matrix_opt_T) @ sim['y'])

fig, axs = plt.subplots(2,8,figsize=(20, 5))
plt.suptitle("Inclination", fontsize=14)
for i, pl in enumerate(planets):
    axs[0][i].set_title(pl)
    pts = sim['y'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = Theta[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
# # %config InlineBackend.figure_format = 'retina'
# # plt.rcParams['font.size'] = 12
# X_full = jnp.concatenate((full_sim['x'], -1j*np.conj(full_sim['x'])), axis=0)
# Phi_full = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ X_full)[:N]
# X_train = jnp.concatenate((train_sim['x'], -1j*np.conj(train_sim['x'])), axis=0)
# Phi_train = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ X_train)[:N]
# # fig, axs = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
# fig, axs = plt.subplots(2, 2, figsize=(15, 7))
# # axs[0][0].plot(full_sim['time'] * TO_YEAR, np.abs(full_sim['x'][0]))
# axs[0][0].plot(train_sim['time'] * TO_YEAR, np.abs(train_sim['x'][0]))
# axs[1][0].plot(full_sim['time'] * TO_YEAR, np.abs(Phi_full[0]))
# axs[1][0].plot(train_sim['time'] * TO_YEAR, np.abs(Phi_train[0]))
# axs[0][0].set_ylim(0, 6.5e-5)
# axs[1][0].set_ylim(0, 6.5e-5)
# axs[0][0].set_title("Mercury Before")
# axs[1][0].set_title("Mercury After")
# # axs[0][0].axvline(train_sim['time'][keep_first] * TO_YEAR, linestyle="--", color="black")
# axs[1][0].axvline(train_sim['time'][keep_first-1] * TO_YEAR, linestyle="--", color="black")
# axs[1][0].set_xlabel("Years")

# # axs[0][1].plot(full_sim['time'] * TO_YEAR, np.abs(full_sim['x'][4]))
# axs[0][1].plot(train_sim['time'] * TO_YEAR, np.abs(train_sim['x'][4]))
# axs[1][1].plot(full_sim['time'] * TO_YEAR, np.abs(Phi_full[4]))
# axs[1][1].plot(train_sim['time'] * TO_YEAR, np.abs(Phi_train[4]))
# axs[0][1].set_ylim(0, 3e-3)
# axs[1][1].set_ylim(0, 3e-3)
# axs[0][1].set_title("Jupiter Before")
# axs[1][1].set_title("Jupiter After")
# # axs[0][1].axvline(train_sim['time'][keep_first] * TO_YEAR, linestyle="--", color="black")
# axs[1][1].axvline(train_sim['time'][keep_first-1] * TO_YEAR, linestyle="--", color="black")
# axs[1][1].set_xlabel("Years")
# # plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=0.2)
# # %config InlineBackend.figure_format = ''
# # plt.rcParams['font.size'] = 10

# # np.savez_compressed("planet_aa",
# #                     full_mercury_before=full_sim['x'][0],
# #                     full_mercury_after=Phi_full[0],
# #                     full_jupiter_before=full_sim['x'][4],
# #                     full_jupiter_after=Phi_full[4],
# #                     full_time=full_sim['time'] * TO_YEAR,
# #                     train_mercury_before=train_sim['x'][0],
# #                     train_mercury_after=Phi_train[0],
# #                     train_jupiter_before=train_sim['x'][4],
# #                     train_jupiter_after=Phi_train[4],
# #                     train_time=train_sim['time'] * TO_YEAR,
# #                     training_time=train_sim['time'][keep_first-1] * TO_YEAR)
# # %%
# def planet_fmft(time, x, N=14, display=False):
#     planet_ecc_fmft = {}
#     for i,pl in enumerate(planets):
#         planet_ecc_fmft[pl] = fmft(time,x[i],N)
#         planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))

#         if display:
#             print("")
#             print(pl)
#             print("-------")
#             for g in planet_e_freqs[:8]:
#                 print("{:+07.3f} \t {:0.8f}".format(g * TO_ARCSEC_PER_YEAR, np.abs(planet_ecc_fmft[pl][g])))
#             # print("s")
#             # print("-------")
#             # for s in planet_inc_freqs[:4]:
#             #     print("{:+07.3f} \t {:0.6f}".format(s * TO_ARCSEC_PER_YEAR, np.abs(planet_inc_fmft[pl][s])))
    
#     return planet_ecc_fmft

# planet_ecc_fmft = planet_fmft(sim['time'], Phi, display=True)
# # %%
# # increase effective simulation cadence

# from multiprocessing import Pool

# full_sim_end_time = full_sim['time'][-1]
# segment_len = int(3e6) / TO_YEAR
# segments = np.arange(0, full_sim_end_time, segment_len)
# pts_per_segment = 4096
# tmp_dir = dataset_path/'tmp4'
# tmp_dir.mkdir(exist_ok=True, parents=True)


# def thread_init(*rest):
#     global sa_
#     sa_ = rb.Simulationarchive(str(dataset_path / "planet_integration.236355012349.500000.sa"))

# def run(segment_start):
#     sim = sa_.getSimulation(segment_start)

#     Tfin_approx = min(segment_start + segment_len, full_sim_end_time) - segment_start
#     total_steps = np.ceil(Tfin_approx / sim.dt)
#     Tfin = total_steps * sim.dt + sim.dt
#     print(Tfin + segment_start, total_steps, int(np.floor(total_steps/pts_per_segment)))

#     sim.save_to_file(str(tmp_dir/f"segment_{segment_start}.sa"), step=int(np.floor(total_steps/pts_per_segment)), delete_file=True)
#     sim.integrate(Tfin + segment_start, exact_finish_time=0)

# with Pool(50, initializer=thread_init) as pool:
#     pool.map(run, segments)
# # %%
# N = full_sim['x'].shape[0]
# X = jnp.concatenate((full_sim['x'], -1j*np.conj(full_sim['x'])), axis=0)
# X_segments = [X[..., (full_sim['time'] >= t_) & (full_sim['time'] < t_ + segment_len)] for t_ in segments]

# def objective(M, J, X):
#     M = jnp.reshape(M, (N*2, N*2))
    
#     symplectic_loss = ((M.T @ J @ M - J)**2).sum()

#     Phi = M.T @ X
#     J_loss = ((jnp.abs(Phi) - jnp.abs(Phi).mean(axis=1)[..., None]) ** 2) / jnp.pow(jnp.abs(Phi).mean(axis=1), 1/5)[..., None]
#     J_loss = J_loss.sum()
#     # J_loss = (J_loss / jnp.pow(jnp.concat((masses, masses))[..., None], 1/4)).sum()
    
#     loss = symplectic_loss + J_loss * 10
#     return loss
# obj_and_grad = jax.jit(jax.value_and_grad(lambda M, X: objective(M, J, X)))

# M_segments = []
# Phi_segments = []

# for i in tqdm(range(len(segments))):
#     segment_start = segments[i]
#     X = X_segments[i]

#     sim_train = load_sim(tmp_dir/f"segment_{segment_start}.sa")
#     train_X = jnp.concatenate((sim_train['x'], -1j*np.conj(sim_train['x'])), axis=0)
#     # initial = ecc_rotation_matrix_opt_T.copy()
#     initial = (M_segments[-1] if M_segments else ecc_rotation_matrix_opt_T).copy()
#     # np.random.seed(i)
#     # initial = ecc_rotation_matrix_opt_T.copy() + np.random.normal(0, 0.1, (N*2, N*2))
#     sol = minimize(obj_and_grad, initial.reshape(-1), args=(train_X,), options={'gtol': 1e-8, 'disp': False}, jac=True)
#     # print(sim_train['time'][0] * TO_YEAR, sim_train['time'][-1] * TO_YEAR)
#     M = sol.x.reshape((N*2, N*2))
#     Phi_segments.append((np.linalg.inv(M) @ X)[:N])
#     M_segments.append(M)
# # %%
# Phi_segments_comb = np.concat(Phi_segments, axis=1)
# M_segments = np.array(M_segments)
# # %%
# x_plt = full_sim['time'] * TO_YEAR / 1e6
# # plt.plot(x_plt, np.abs(full_sim['x'][0]), label="orig")
# # plt.plot(x_plt, np.abs(Phi_full[0]), label="train whole")
# # plt.plot(x_plt, np.abs(Phi_segments_comb[0]), label="train 2 Myr")

# prev_rng = [-1]
# for seg in Phi_segments:
#     rng = np.arange(prev_rng[-1]+1, prev_rng[-1]+len(seg[0])+1)
#     plt.plot(full_sim['time'][rng] * TO_YEAR/1e6, np.abs(seg[0]))
#     prev_rng = rng

# # plt.xlim(0, 20)

# plt.xlabel("Myr")
# plt.legend()
# plt.show()
# # %%
# M_segments_flat = M_segments.reshape(len(segments), -1)
# plt.imshow(M_segments_flat.T)
# # %%
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# sc.fit(M_segments_flat)

# pca = PCA(n_components=5)
# pca.fit(sc.transform(M_segments_flat))
# plt.plot(pca.transform(sc.transform(M_segments_flat)))
# # %%
# def run(segment_start):
#     sim = load_sim(tmp_dir/f"segment_{segment_start}.sa")
#     return sim, planet_fmft(sim['time'], sim['x'])

# with Pool(50) as pool:
#     sims, fmfts = zip(*list(tqdm(pool.imap(run, segments), total=len(segments))))
# # %%
# top_freqs = np.array([5.11, 4.24, 7.31, 6.17]) / TO_ARCSEC_PER_YEAR
# amp = []
# freq = []
# a = []
# for i in range(len(fmfts)):
#     amp.append([])
#     freq.append([])
#     a.append(np.mean(sims[i]['a'][0]))

#     for f_ in top_freqs:
#         f, a_ = closest_key_entry(fmfts[i]['Mercury'],f_)
#         freq[-1].append(f)
#         amp[-1].append(a_)

# amp = np.array(amp)
# freq = np.array(freq) * TO_ARCSEC_PER_YEAR
# # %%
# for i in range(freq.shape[0]):
#     plt.scatter(np.ones_like(top_freqs) * i, freq[0], alpha=np.abs(amp[0])/np.linalg.norm(np.abs(amp[0])))
# # %%
# plt.plot(freq[:,0])
# # plt.ylim(4.5, 5.5)
# # %%
# plt.scatter(freq[:,0], freq[:,1])
# # plt.scatter(a, freq[:,1])
# # plt.xlim(4.5, 5.5)
# # %%
# from scipy.signal import ShortTimeFFT
# from scipy.signal.windows import gaussian

# N = full_sim['time'].shape[0]
# T_x = np.mean(np.gradient(full_sim['time'])) * TO_YEAR
# w = gaussian(500, std=8, sym=True)
# SFT = ShortTimeFFT(w, hop=10, fs=1/T_x, scale_to='magnitude', fft_mode='centered')
# Sx = SFT.stft(full_sim['x'][0])

# extent = np.array(SFT.extent(N))
# extent[:2] = extent[:2] * 1e-6
# extent[2:] = extent[2:] * TO_ARCSEC_PER_YEAR

# plt.imshow(abs(Sx), origin='lower', aspect='auto', extent=extent, cmap='viridis')
# plt.ylim(bottom=0)

# plt.xlabel("Myr")
# plt.ylabel("''/yr")
# %%
planet_ecc_fmft = {}
planet_inc_fmft = {}
for i,pl in enumerate(planets):
    planet_ecc_fmft[pl] = fmft(sim['time'],Phi[i],14)
    planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))

    planet_inc_fmft[pl] = fmft(sim['time'],Theta[i],14)
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
    
# %%
g_vec = np.zeros(8)
s_vec = np.zeros(8)

for i in range(len(planets)):
    freqs = list(planet_ecc_fmft[planets[i]].keys())
    for f in freqs:
        if (np.abs((g_vec[:i] / f) - 1) < 1e-4).any():
            continue
        g_vec[i] = f
        break

for i in range(len(planets)):
    freqs = list(planet_inc_fmft[planets[i]].keys())
    for f in freqs:
        if i != np.argmax(masses) and np.abs(f*TO_ARCSEC_PER_YEAR) < 0.001:
            continue
        if (np.abs((s_vec[:i] / f) - 1) < 1e-4).any():
            continue
        s_vec[i] = f
        break

g_vec[0] = list(planet_ecc_fmft['Mercury'].keys())[0]
g_vec[1] = list(planet_ecc_fmft['Venus'].keys())[0]
g_vec[2] = list(planet_ecc_fmft['Earth'].keys())[0]
g_vec[3] = list(planet_ecc_fmft['Mars'].keys())[0]
g_vec[4] = list(planet_ecc_fmft['Jupiter'].keys())[0]
g_vec[5] = list(planet_ecc_fmft['Saturn'].keys())[0]
g_vec[6] = list(planet_ecc_fmft['Uranus'].keys())[0]
g_vec[7] = list(planet_ecc_fmft['Neptune'].keys())[0]

s_vec[0] = list(planet_inc_fmft['Mercury'].keys())[0]
s_vec[1] = list(planet_inc_fmft['Venus'].keys())[0]
s_vec[2] = list(planet_inc_fmft['Earth'].keys())[0]
s_vec[3] = list(planet_inc_fmft['Mars'].keys())[0]
s_vec[4] = list(planet_inc_fmft['Jupiter'].keys())[0]
s_vec[5] = list(planet_inc_fmft['Saturn'].keys())[0]
s_vec[6] = list(planet_inc_fmft['Uranus'].keys())[0]
s_vec[7] = list(planet_inc_fmft['Neptune'].keys())[0]

g_amp = np.zeros(8, dtype=np.complex128)
s_amp = np.zeros(8, dtype=np.complex128)

g_amp[0] = planet_ecc_fmft['Mercury'][g_vec[0]]
g_amp[1] = planet_ecc_fmft['Venus'][g_vec[1]]
g_amp[2] = planet_ecc_fmft['Earth'][g_vec[2]]
g_amp[3] = planet_ecc_fmft['Mars'][g_vec[3]]
g_amp[4] = planet_ecc_fmft['Jupiter'][g_vec[4]]
g_amp[5] = planet_ecc_fmft['Saturn'][g_vec[5]]
g_amp[6] = planet_ecc_fmft['Uranus'][g_vec[6]]
g_amp[7] = planet_ecc_fmft['Neptune'][g_vec[7]]

s_amp[0] = planet_inc_fmft['Mercury'][s_vec[0]]
s_amp[1] = planet_inc_fmft['Venus'][s_vec[1]]
s_amp[2] = planet_inc_fmft['Earth'][s_vec[2]]
s_amp[3] = planet_inc_fmft['Mars'][s_vec[3]]
s_amp[4] = planet_inc_fmft['Jupiter'][s_vec[4]]
s_amp[5] = planet_inc_fmft['Saturn'][s_vec[5]]
s_amp[6] = planet_inc_fmft['Uranus'][s_vec[6]]
s_amp[7] = planet_inc_fmft['Neptune'][s_vec[7]]

omega_vec = np.concat([g_vec, s_vec])
omega_amp = np.concat([g_amp, s_amp])

s_conserved_idx = np.argmin(np.abs(omega_vec))

print(np.array([5.535, 7.437, 17.357, 17.905, 4.257, 28.77, 3.088, 0.671] + [-5.624, -7.082, -18.837, -17.749, 0.0, -26.348, -2.993, -0.692]))
print((omega_vec * TO_ARCSEC_PER_YEAR).round(3))
print(omega_amp)
# %%
base_planet_list = planets
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
planet_fmft = get_planet_fmft(psi_planet_list, sim['time'], Psi, N=14, display=True)
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
                omega_N_exists = False
                to_del = []
                for old_k,(_, old_omega, old_err) in comb.items():
                    if old_omega == omega_N:
                        omega_N_exists = True
                        if old_err > omega_pct_error:
                            # del comb[old_k]
                            to_del.append(old_k)
                            omega_N_exists = False
                for d in to_del:
                    del comb[d]
                if not omega_N_exists:
                    comb[tuple(k)] = (amp, omega_N, omega_pct_error)
                    if display: print (k,"\t{:+07.3f}\t{:.1g},\t{:.1g}".format(omega*TO_ARCSEC_PER_YEAR,omega_pct_error,np.abs(amp)))
        combs.append(comb)
    return combs

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
iterations = [1,3]
# iterations = [1,3,5]
# iterations = [1,3,5,7]
for i,order in enumerate(iterations):
    print("#"*10, f"ITERATION {i+1} - ORDER {order}", "#"*10)
    last_x_val = apply_sequential_transforms(x_val, trans_fns)
    last_fmft = get_planet_fmft(psi_planet_list, sim['time'], last_x_val, 14, display=False)
    combs = get_combs(order, last_fmft, psi_planet_list, omega_vec, display=True, include_negative=False, omega_pct_thresh=5e-5)

    x_bar_i = [sympy.Symbol(f"\\bar X^{{({i+1})}}_"+str(j)) for j in range(N*2)]

    # loop through each object
    for j in range(N*2):
        # to first order the coordinate is the original coordinate
        x_bar_i_j = x_bars[-1][j]
        # correct for each combination
        for k,(amp, omega, _) in combs[j].items():
            term = amp
            # loop through each object
            delta = 0
            for k_idx in range(N*2):
                # add each object the correct number of times
                for l in range(np.abs(k[k_idx])):
                    term *= x_bars[-1][k_idx]/omega_amp[k_idx] if k[k_idx] > 0 else x_bars[-1][k_idx].conjugate()/np.conj(omega_amp[k_idx])
            x_bar_i_j -= term
        subs[x_bar_i[j]] = x_bar_i_j
    x_bars.append(x_bar_i)
    trans_fns.append(eval_transform(x_bars, subs))

Psi_trans = apply_sequential_transforms(x_val, trans_fns)
# %%
b, a = scipy.signal.butter(10, 100, 'low', fs=fs_arcsec_per_yr) # type: ignore[reportUnknownVariableType]
# Psi_filt = scipy.signal.lfilter(b, a, Psi_trans)
Psi_filt = Psi_trans
# plt.plot(np.real(Psi_filt[3]), np.imag(Psi_filt[3]))
# %%
new_planet_fmft = get_planet_fmft(psi_planet_list, sim['time'], Psi_trans, N=14, display=True, compareto=last_fmft)
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
fig, axs = plt.subplots(2, N,figsize=(20,5))
for i, pl in enumerate(planets):
    axs[0][i].set_title(pl)
    # axs[0][i].plot(sim['time'][100:], sim['x'][i][100:] * np.conj(sim['x'][i])[100:], label='Original Coordinate')
    # axs[1][i].plot(sim['time'][100:], sim['y'][i][100:] * np.conj(sim['y'][i])[100:])

    axs[0][i].plot(sim['time'][100:], Psi[i][100:] * np.conj(Psi[i])[100:], label='After Rotation')
    axs[1][i].plot(sim['time'][100:], Psi[i+N][100:] * np.conj(Psi[i+N])[100:])

    axs[0][i].plot(sim['time'][100:], Psi_filt[i][100:] * np.conj(Psi_filt[i])[100:], label='After lasers')
    axs[1][i].plot(sim['time'][100:], Psi_filt[i+N][100:] * np.conj(Psi_filt[i+N])[100:])

    
    axs[0][i].set_ylim(-axs[0][i].get_ylim()[1] / 10, axs[0][i].get_ylim()[1] * 1.5)
    axs[1][i].set_ylim(-axs[1][i].get_ylim()[1] / 10, axs[1][i].get_ylim()[1] * 1.5)
axs[0][0].set_ylabel("Eccentricity")
axs[1][0].set_ylabel("Inclination")
axs[0][0].legend()
plt.show()
# %%
# script X matching action variable in Mogavero & Laskar (2023)
# sX = np.real(sim['x'] * np.conj(sim['x']))
sX = np.real(Psi[:N] * np.conj(Psi[:N]))
# sX = np.real(Psi_filt[:N] * np.conj(Psi_filt[:N]))
# script Psi
# sPsi = np.real(sim['y'] * np.conj(sim['y']))
sPsi = np.real(Psi[N:] * np.conj(Psi[N:]))
# sPsi = np.real(Psi_filt[N:] * np.conj(Psi_filt[N:]))
# %%
gamma_0 = np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
C_ecc = gamma_0[:N] @ sX + gamma_0[N:] @ sPsi

gamma_1 = np.array([0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0])
C_inc = gamma_1[:N] @ sX + gamma_1[N:] @ sPsi
C_0 = (C_inc + C_ecc)[-1]

C_inc_hat = C_inc / (np.linalg.norm(gamma_1) * C_0)

gamma_2 = np.array([0,0,-1,-1,0,0,0,0,1,1,2,2,0,0,0,0])
C_2 = gamma_2[:N] @ sX + gamma_2[N:] @ sPsi
C_2_hat = C_2 / (np.linalg.norm(gamma_2) * C_0)

t = sim['time'] / (2*np.pi*1e6)

plt.plot(t, sX[0]/C_0 - (sX[0]/C_0).mean(), label=r"$\hat \mathcal{X}_1$", c="tab:blue")
plt.plot(t, sPsi[2]/C_0 - (sPsi[2]/C_0).mean(), label=r"$\hat \mathcal{\Psi}_3$", c="tab:cyan")
plt.plot(t, C_inc_hat - C_inc_hat.mean(), label=r"$C_{inc}$", c="tab:orange")
plt.plot(t, C_2_hat - C_2_hat.mean(), label=r"$\hat C_{2}$", c="tab:red")
plt.xlim(left=t[100], right=t[-1])
plt.legend()
plt.xlabel("Myr")
plt.show()
# %%
def objective(A, X):
    int_loss = ((A - A.round())**2).sum()
    # non_zero_loss = -jnp.linalg.norm(A)
    non_zero_loss = - (jnp.abs(A).sum() / jnp.abs(A).max())
    J = A @ X #/ jnp.linalg.norm(A)
    J = J / C_0

    J_approx = jnp.abs(J).mean()
    J_loss = ((jnp.abs(J) - J_approx) ** 2)
    J_loss = J_loss.sum()

    return J_loss + int_loss*5e-1 + non_zero_loss*1e-3
obj_no_grad = jax.jit(lambda A: objective(A, jnp.concat((sX, sPsi))))
obj_and_grad = jax.jit(jax.value_and_grad(lambda A: objective(A, jnp.concat((sX, sPsi)))))

lw=[-2]*(N*2)
up=[3]*(N*2)
sol = dual_annealing(obj_no_grad, bounds=list(zip(lw, up)))
# sol = minimize(obj_and_grad, jnp.ones(N*2)*0.5, options={'gtol': 1e-8, 'disp': True}, jac=True)
# sol = minimize(obj_no_grad, jnp.array(gamma_2), options={'gtol': 1e-8, 'disp': True})

gamma_n = sol.x.round()
print(gamma_n)

# plt.plot(t, sX[0]/C_0 - (sX[0]/C_0).mean(), label=r"$\hat \mathcal{X}_1$", c="tab:blue")
plt.plot(t, sPsi[2]/C_0 - (sPsi[2]/C_0).mean(), label=r"$\hat \mathcal{\Psi}_3$", c="tab:cyan")
# plt.plot(t, C_inc_hat - C_inc_hat.mean(), label=r"$C_{inc}$", c="tab:orange")
# plt.plot(t, C_2_hat - C_2_hat.mean(), label=r"$\hat C_{2}$", c="tab:red")

C_opt = gamma_n @ np.concat((sX, sPsi))
C_opt_hat = C_opt / (np.linalg.norm(gamma_n) * C_0)
plt.plot(t, C_opt_hat - C_opt_hat.mean(), label=r"$\hat C_{opt}$", c="tab:purple")
plt.xlim(left=t[100], right=t[-1])
plt.legend()
plt.xlabel("Myr")
plt.show()
# %%
