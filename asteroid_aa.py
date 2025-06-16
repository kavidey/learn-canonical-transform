# %%
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jnr
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
from flax import nnx
import optax
import orbax.checkpoint as ocp
from nn.sympnet.la import LA_Layer
from nn.utils import get_pq, get_x

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
desns = sorted(desns)
# %%
rb_sim = rb.Simulation(str(dataset_path/'planets.bin'))
masses = np.array(list(map(lambda ps: ps.m, rb_sim.particles))[1:] + [1e-12])

def load_sim(desn):
    results = get_simarchive_integration_results(str(dataset_path / f"{'_'.join(sims[0].stem.split('_')[:-1])}_{desn}.sa"), coordinates='heliocentric')
    
    results['X'] = np.sqrt(2*(1-np.sqrt(1-results['e']**2))) * np.exp(1j * results['pomega'])
    results['Y'] = (1-results['e']**2)**(0.25) * np.sin(0.5 * results['inc'] ) * np.exp(1j * results['Omega']) 

    prefactor = np.sqrt(masses)[..., None].repeat(results['X'].shape[1], axis=-1) * np.power(rb_sim.G * rb_sim.particles[0].m * results['a'], 1/4)
    results['G'] = prefactor * results['X']
    results['F'] = prefactor * results['Y']

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
ecc_rotation_matrix_T = ecc_rotation_matrix_T / np.linalg.norm(ecc_rotation_matrix_T, axis=0)
print(np.linalg.det(ecc_rotation_matrix_T))

with np.printoptions(suppress=True, precision=4):
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
transformedY = (sim['F'].T @ inc_rotation_matrix_T.T).T

fig, axs = plt.subplots(2,5,figsize=(15, 5))
for i in range(5):
    pts = sim['F'][i]
    axs[0][i].plot(np.real(pts), np.imag(pts))
    axs[0][i].set_aspect('equal')
    pts = transformedY[i]
    axs[1][i].plot(np.real(pts), np.imag(pts))
    axs[1][i].set_aspect('equal')
# %%
asteroid_phi = Phi[-1][:50]
plt.plot(np.angle(asteroid_phi), np.abs(asteroid_phi))
# %%
angle = np.angle(asteroid_phi)
plt.plot(sim['time'][:100], jnp.gradient(jnp.sin(angle)))
plt.plot(sim['time'][:100], jnp.cos(angle) * (1/(1.5 * jnp.pi)))
# %%
train_steps = 10000
eval_every = 200
batch_size = 1024

data = np.concatenate((np.abs(Phi), np.angle(Phi)), axis=0).T
# data = np.concatenate((np.real(Phi), np.imag(Phi)), axis=0).T

train_ds = tf.data.Dataset.from_tensor_slices(data)
train_ds = train_ds.repeat()#.shuffle(256)
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
# %%
class SympTransform(nnx.Module):
    def __init__(self, N: int, layers, sublayers, *, rngs: nnx.Rngs, dt=0.1, epsilon=0.1):
        self.N = N
        self.dt = dt
        self.epsilon = epsilon
        self.layers = [LA_Layer(N, sublayers, rngs=rngs) for _ in range(layers)]

    def __call__(self, x):
        x_orig = x
        h = self.dt * jnp.ones_like(x[..., -1:])
        for layer in self.layers:
            x = layer(x, h)
        return x_orig + self.epsilon * x
# %%
@nnx.jit
def train_step(
    model: SympTransform,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch
):
    def loss_fn(model: SympTransform, batch):
        Y = model(batch)
        
        p, q = get_pq(Y, model.N)

        # J = jnp.sqrt(p ** 2 + q ** 2)
        J = p
        J_loss = (jnp.gradient(J, axis=-2) ** 2).sum()
        # J_loss = ((p - jnp.mean(p)) ** 2).mean()
        J_magnitude_loss = 1/(jnp.linalg.norm(J, axis=-2) + 0.001).sum()

        # phi = jnp.atan2(p, q)
        phi = q
        omega = jnp.gradient(jnp.sin(phi), axis=-2)
        omega_loss = ((omega - jnp.cos(phi) * (1/(1.5 * jnp.pi))) ** 2).sum()

        loss = J_loss + omega_loss
        # loss = J_loss + J_magnitude_loss

        return loss, (p, q)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(model, batch)
    metrics.update(loss=loss)
    optimizer.update(grads)

    return aux
# %%
model = SympTransform(int(data.shape[-1]/2), 5, 3, rngs=nnx.Rngs(0))

learning_rate = 0.005
momentum = 0.9
opt = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))

metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
# %%
step_list = []
metric_history = {"train_loss": []}

for step, batch in enumerate(train_ds.as_numpy_iterator()):
    model_meta = train_step(model, opt, metrics, batch)

    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
        for metric, value in metrics.compute().items():
            metric_history[f"train_{metric}"].append(value)
        metrics.reset()

        step_list.append(step)

        clear_output(wait=True)

        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
        ax1.set_title("Loss")
        ax1.plot(step_list, metric_history["train_loss"])
        plt.show()
# %%
obj_idx = 4
test_data = data[:2000]
p, q = get_pq(test_data, model.N)
x = p[...,obj_idx] * np.exp(1j * q[...,obj_idx])
# x = p[...,-1] + 1j * q[...,-1]

P, Q = get_pq(model(test_data), model.N)
X = P[...,obj_idx] * np.exp(1j * Q[...,obj_idx])
# X = P[..., -1] + 1j * Q[...,-1]

fig, (axs) = plt.subplots(3, 2, figsize=(15, 15))
axs[0, 0].plot(jnp.abs(x))
axs[0, 1].plot(jnp.abs(X))

axs[1, 0].plot(jnp.gradient(jnp.sin(jnp.angle(x))))
axs[1, 0].plot(jnp.cos(jnp.angle(x)) / (1.5 * jnp.pi))
axs[1, 1].plot(jnp.gradient(jnp.sin(jnp.angle(X))))
axs[1, 1].plot(jnp.cos(jnp.angle(X)) / (1.5 * jnp.pi))

axs[2, 0].plot(jnp.real(x), jnp.imag(x))
axs[2, 0].set_aspect('equal')
axs[2, 1].plot(jnp.real(X), jnp.imag(X))
axs[2, 1].set_aspect('equal')
# %%
plt.plot(jnp.angle(x), jnp.abs(x))
plt.plot(jnp.angle(X), jnp.abs(X))
# %%
