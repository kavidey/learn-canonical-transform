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
full_sim = load_sim(dataset_path / "planet_integration.236355012349.500000.sa")
print(full_sim['time'].shape, full_sim['time'][-1] * TO_YEAR)
for key, val in full_sim.items():
    full_sim[key] = val[..., :24081]
print(full_sim['time'].shape, full_sim['time'][-1] * TO_YEAR)
# %%
# train_sim = load_sim(dataset_path / "planet_integration.59088753087.500000.sa")
train_sim = load_sim(dataset_path / "planet_integration.590887530.200000.sa")
# train_sim = load_sim(dataset_path / "planet_integration.sa")
print(train_sim['time'].shape, train_sim['time'][-1] * TO_YEAR)
# keep_first = int(train_sim['time'].shape[0]*0.9)
keep_first = int(50e3)
print(train_sim['time'][keep_first] * TO_YEAR)
sim = {}
for key, val in train_sim.items():
    sim[key] = val[..., :keep_first]
train_sim = sim
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
    J_loss = ((jnp.abs(Phi) - jnp.abs(Phi).mean(axis=1)[..., None]) ** 2) / jnp.pow(jnp.abs(Phi).mean(axis=1), 1/5)[..., None]
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
X_full = jnp.concatenate((full_sim['x'], -1j*np.conj(full_sim['x'])), axis=0)
Phi_full = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ X_full)[:N]
X_train = jnp.concatenate((train_sim['x'], -1j*np.conj(train_sim['x'])), axis=0)
Phi_train = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ X_train)[:N]
# fig, axs = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
fig, axs = plt.subplots(2, 2, figsize=(15, 7))
# axs[0][0].plot(full_sim['time'] * TO_YEAR, np.abs(full_sim['x'][0]))
axs[0][0].plot(train_sim['time'] * TO_YEAR, np.abs(train_sim['x'][0]))
axs[1][0].plot(full_sim['time'] * TO_YEAR, np.abs(Phi_full[0]))
axs[1][0].plot(train_sim['time'] * TO_YEAR, np.abs(Phi_train[0]))
axs[0][0].set_ylim(0, 6.5e-5)
axs[1][0].set_ylim(0, 6.5e-5)
axs[0][0].set_title("Mercury Before")
axs[1][0].set_title("Mercury After")
# axs[0][0].axvline(train_sim['time'][keep_first] * TO_YEAR, linestyle="--", color="black")
axs[1][0].axvline(train_sim['time'][keep_first-1] * TO_YEAR, linestyle="--", color="black")
axs[1][0].set_xlabel("Years")

# axs[0][1].plot(full_sim['time'] * TO_YEAR, np.abs(full_sim['x'][4]))
axs[0][1].plot(train_sim['time'] * TO_YEAR, np.abs(train_sim['x'][4]))
axs[1][1].plot(full_sim['time'] * TO_YEAR, np.abs(Phi_full[4]))
axs[1][1].plot(train_sim['time'] * TO_YEAR, np.abs(Phi_train[4]))
axs[0][1].set_ylim(0, 3e-3)
axs[1][1].set_ylim(0, 3e-3)
axs[0][1].set_title("Jupiter Before")
axs[1][1].set_title("Jupiter After")
# axs[0][1].axvline(train_sim['time'][keep_first] * TO_YEAR, linestyle="--", color="black")
axs[1][1].axvline(train_sim['time'][keep_first-1] * TO_YEAR, linestyle="--", color="black")
axs[1][1].set_xlabel("Years")
# plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=0.2)
# %config InlineBackend.figure_format = ''
# plt.rcParams['font.size'] = 10

# np.savez_compressed("planet_aa",
#                     full_mercury_before=full_sim['x'][0],
#                     full_mercury_after=Phi_full[0],
#                     full_jupiter_before=full_sim['x'][4],
#                     full_jupiter_after=Phi_full[4],
#                     full_time=full_sim['time'] * TO_YEAR,
#                     train_mercury_before=train_sim['x'][0],
#                     train_mercury_after=Phi_train[0],
#                     train_jupiter_before=train_sim['x'][4],
#                     train_jupiter_after=Phi_train[4],
#                     train_time=train_sim['time'] * TO_YEAR,
#                     training_time=train_sim['time'][keep_first-1] * TO_YEAR)
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
# increase effective simulation cadence

from multiprocessing import Pool

full_sim_end_time = full_sim['time'][-1]
segment_len = int(10e6) / TO_YEAR
segments = np.arange(0, full_sim_end_time, segment_len)
pts_per_segment = 4096
tmp_dir = dataset_path/'tmp3'
tmp_dir.mkdir(exist_ok=True, parents=True)


def thread_init(*rest):
    global sa_
    sa_ = rb.Simulationarchive(str(dataset_path / "planet_integration.236355012349.500000.sa"))

def run(segment_start):
    sim = sa_.getSimulation(segment_start)

    Tfin_approx = min(segment_start + segment_len, full_sim_end_time) - segment_start
    total_steps = np.ceil(Tfin_approx / sim.dt)
    Tfin = total_steps * sim.dt + sim.dt
    print(Tfin + segment_start, total_steps, int(np.floor(total_steps/pts_per_segment)))

    sim.save_to_file(str(tmp_dir/f"segment_{segment_start}.sa"), step=int(np.floor(total_steps/pts_per_segment)), delete_file=True)
    sim.integrate(Tfin + segment_start, exact_finish_time=0)

with Pool(50, initializer=thread_init) as pool:
    pool.map(run, segments)
# %%
N = full_sim['x'].shape[0]
X = jnp.concatenate((full_sim['x'], -1j*np.conj(full_sim['x'])), axis=0)
X_segments = [X[..., (full_sim['time'] >= t_) & (full_sim['time'] < t_ + segment_len)] for t_ in segments]

def objective(M, J, X):
    M = jnp.reshape(M, (N*2, N*2))
    
    symplectic_loss = ((M.T @ J @ M - J)**2).sum()

    Phi = M.T @ X
    J_loss = ((jnp.abs(Phi) - jnp.abs(Phi).mean(axis=1)[..., None]) ** 2) / jnp.pow(jnp.abs(Phi).mean(axis=1), 1/5)[..., None]
    J_loss = J_loss.sum()
    # J_loss = (J_loss / jnp.pow(jnp.concat((masses, masses))[..., None], 1/4)).sum()
    
    loss = symplectic_loss + J_loss * 10
    return loss
obj_and_grad = jax.jit(jax.value_and_grad(lambda M, X: objective(M, J, X)))

M_segments = []
Phi_segments = []

for i in tqdm(range(len(segments))):
    segment_start = segments[i]
    X = X_segments[i]

    sim_train = load_sim(tmp_dir/f"segment_{segment_start}.sa")
    train_X = jnp.concatenate((sim_train['x'], -1j*np.conj(sim_train['x'])), axis=0)
    # initial = ecc_rotation_matrix_opt_T.copy()
    initial = (M_segments[-1] if M_segments else ecc_rotation_matrix_opt_T).copy()
    # np.random.seed(i)
    # initial = ecc_rotation_matrix_opt_T.copy() + np.random.normal(0, 0.1, (N*2, N*2))
    sol = minimize(obj_and_grad, initial.reshape(-1), args=(train_X,), options={'gtol': 1e-8, 'disp': False}, jac=True)
    # print(sim_train['time'][0] * TO_YEAR, sim_train['time'][-1] * TO_YEAR)
    M = sol.x.reshape((N*2, N*2))
    Phi_segments.append((np.linalg.inv(M) @ X)[:N])
    M_segments.append(M)
# %%
Phi_segments_comb = np.concat(Phi_segments, axis=1)
M_segments = np.array(M_segments)
# %%
x_plt = full_sim['time'] * TO_YEAR / 1e6
plt.plot(x_plt, np.abs(full_sim['x'][0]), label="orig")
plt.plot(x_plt, np.abs(Phi_full[0]), label="train whole")
plt.plot(x_plt, np.abs(Phi_segments_comb[0]), label="train 10 Myr")

# prev_rng = [-1]
# for seg in Phi_segments:
#     rng = np.arange(prev_rng[-1]+1, prev_rng[-1]+len(seg[0])+1)
#     plt.plot(full_sim['time'][rng] * TO_YEAR/1e6, np.abs(seg[0]))
#     prev_rng = rng

plt.xlim(0, 20)

plt.xlabel("Myr")
plt.legend()
plt.show()
# %%
M_segments_flat = M_segments.reshape(len(segments), -1)
plt.imshow(M_segments_flat.T)
# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(M_segments_flat)

pca = PCA(n_components=5)
pca.fit(sc.transform(M_segments_flat))
plt.plot(pca.transform(sc.transform(M_segments_flat)))
# %%
