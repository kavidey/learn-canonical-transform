# %% [markdown]
# ### Imports
# %%
# from pysr import PySRRegressor

from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jnr
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
import tensorflow_datasets as tfds

from data.pendulum import get_dataset, hamiltonian_fn

from flax import nnx
import optax
import orbax.checkpoint as ocp

from nn.sympnet.p import P_Layer
from nn.sympnet.la import LA_Layer

seed = 42
data_dir = Path('datasets')
# %% [markdown]
# ### Setup Dataset
# %%
from scipy.special import ellipk, ellipe, ellipkinc

# Define 𝜅
def compute_kappa(E, F):
    return np.sqrt((1+E/F)/2)

# Define η
def compute_eta(kappa, phi):
    return np.real(np.emath.arcsin(np.sin(0.5 * phi) / kappa))

# Define the transformation into action (J)
def J_transform(G, F, E):
    kappa = compute_kappa(E, F)
    R = np.sqrt(F/G)
    
    if kappa < 1:
        J = R * (8/np.pi) * (ellipe(kappa**2) - (1-kappa**2)*ellipk(kappa**2))
    
    else:
        J = R * 8/np.pi * 1/2 * kappa * ellipe(kappa * (-2))

    return J

# Define the transformation into angle (phi)
def phi_transform(theta, E, F):
    kappa = compute_kappa(E, F)
    eta = compute_eta(kappa, theta)

    if kappa < 1:
        phi = (np.pi/2) * (1/ellipk(kappa**2) * ellipkinc(eta, kappa**2))
    
    else:
        phi = np.pi/2 * 2 * ellipk(kappa ** (-2)) ** (-1) * ellipkinc(0.5 * theta, kappa ** (-2))

    return phi
# %%
np.random.seed(seed)
F = 3
G = 2
omega_0 = np.sqrt(F * G)
data_dir.mkdir(parents=True, exist_ok=True)
# data = get_dataset(samples=1024, seed=seed, noise_std=0.005)
# np.save(data_dir / "pendulum.npy", data)
N = 1
# %%
data = np.load(data_dir / "pendulum.npy", allow_pickle=True).flat[0]
train_x = data["x"]
# train_x = np.concat((train_x, (train_x[:,:,0] ** 2 + train_x[:,:,1] ** 2)[..., None]), axis=-1)
x = (train_x[:, :, 0] + 1j * train_x[:, :, 1])[..., None]
train_J = np.abs(x)
train_phi = np.angle(x)
# train_phi = np.gradient(train_phi, axis=-2)
# phi_median = np.median(train_phi, axis=-2)
# np.putmask(train_phi, np.abs(train_phi) > 2.85, phi_median.repeat(30, axis=-1)[..., None])
# test_x = data['test_x']

train_dxdt = data["dx"]
# test_dxdt = data['test_dx']

train_steps = 10000
eval_every = 200
batch_size = 64

train_ds = tf.data.Dataset.from_tensor_slices((train_phi, train_J))
train_ds = train_ds.repeat().shuffle(256)
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
# %%
fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))

for i in range(len(train_x)):
    x = data["x"][i][:, 0] + 1j * data["x"][i][:, 1]
    ax1.plot(jnp.real(x), jnp.imag(x))

ax1.set_aspect('equal')
# %%
def grad_ignore_jump(x):
    xp = jnp.gradient(x)
    median = jnp.median(xp)
    xp = xp.at[jnp.abs(xp) > jnp.pi * 0.8].set(median)

    return xp

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
for i in range(5):
    H = hamiltonian_fn(data["x"][i].T)[0]
    true_J = jnp.array(list(map(lambda H: J_transform(G, F, H), H)))
    true_phi = jnp.array(phi_transform(data["x"][i].T[0], H.mean(), F))
    
    true_phi_dot = grad_ignore_jump(true_phi)
    true_phi = jnp.cumsum(-jnp.abs(true_phi_dot)) + train_phi[i][0]#+ true_phi[0]
    
    kappa = compute_kappa(H.mean(), 3)
    true_omega = (jnp.pi/2) * 1/ellipk(kappa**2) * omega_0


    ax1.plot(train_J[i])
    ax1.plot(true_J, c="grey", linestyle="--")
    # ax1.set_title("$J$ vs $p$")
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$J$")
    
    ax2.plot(train_phi[i])
    # ax2.set_xlabel(r"$t$")
    # ax2.set_ylabel(r"$\phi$")
    ax2.plot(true_phi, c="grey", linestyle="--")

    print(jnp.abs(true_phi_dot)[0], true_omega)

# remove duplicate legend entries
# ax1.legend(*[*zip(*{l:h for h,l in zip(*ax1.get_legend_handles_labels())}.items())][::-1])
# ax2.legend(*[*zip(*{l:h for h,l in zip(*ax2.get_legend_handles_labels())}.items())][::-1])
# %% [markdown]
# ### Initialize Model
# %%
class Block(nnx.Module):
    def __init__(self, din, dout, *, initializer, activation, rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs, kernel_init=initializer)
        # self.bn = nnx.BatchNorm(dout, rngs=rngs)
        self.activation = activation

    def __call__(self, x):
        # return self.activation(self.bn(self.linear(x)))
        return self.activation(self.linear(x))


class MLP(nnx.Module):
    def __init__(
        self, dims, *, initializer, activation, rngs
    ):  # explicit RNG threading
        self.blocks = []
        for i in range(len(dims) - 1):
            self.blocks.append(
                Block(
                    din=dims[i],
                    dout=dims[i + 1],
                    initializer=initializer,
                    activation=activation,
                    rngs=rngs,
                )
            )

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x


orthogonal_initializer = nnx.initializers.orthogonal()


class MotionConstant(nnx.Module):
    def __init__(self, N: int, *, rngs: nnx.Rngs, dt=0.1):
        self.N = N
        self.dt = dt
        # self.mlp = MLP(
        #     [2, 10, 15, 5, 1],
        #     initializer=orthogonal_initializer,
        #     activation=nnx.gelu,
        #     rngs=rngs,
        # )
        # self.layers = [P_Layer(N, rngs=rngs)] * 5
        self.layers = [LA_Layer(N, 3, rngs=rngs) for _ in range(5)]


    def __call__(self, x):
        h = self.dt * jnp.ones_like(x[..., -1:])
        # print(x.shape)
        # apply_layer = lambda x, layer: (layer(x, h), None)
        # # return jax.lax.scan(apply_layer, x, self.layers)
        # return nnx.scan(apply_layer, length=len(self.layers))(x, self.layers)
        for layer in self.layers:
            x = layer(x, h)
        return x[..., :N]


class GeneratingFunction(nnx.Module):

    def __init__(self, N: int, rngs: nnx.Rngs, epsilon=1.0):
        self.mlp = MLP(
            [2*N, 10, 15, 5, 1],
            initializer=orthogonal_initializer,
            activation=nnx.gelu,
            rngs=rngs,
        )
        self.N = N
        self.epsilon=epsilon

    def __call__(self, q, J):
        return jnp.sum(q * J, axis=-1)[..., None] + self.epsilon * self.mlp(jnp.concat((q, J), axis=-1))
    
    def time_derivative(self, q, J):
        dF2dq, dF2dJ = nnx.grad(lambda q, J: self(q, J).sum(), argnums=[0,1])(q, J)

        return dF2dq, dF2dJ
# %% [markdown]
# ### Training Code
# %%
@nnx.jit
def mc_train_step(
    model: MotionConstant,
    generating_function: GeneratingFunction,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch,
):
    def loss_fn(
        motion_constant: MotionConstant, generating_function: GeneratingFunction, batch
    ):
        q, p = batch

        J = motion_constant(jnp.concat((q, p), axis=-1))

        const_loss = jnp.pow(jnp.gradient(J, axis=-2), 2).sum()
        # const_loss = jnp.pow(J - J.mean(axis=-2)[..., None, :], 2).sum()
        spread_loss = -jnp.std(jnp.mean(J, axis=-2), axis=0).sum()
        
        # dF2 = nnx.grad(lambda x: generating_function(x).sum())(jnp.concat((q, J), axis=-1))
        # dF2dq = dF2[..., :N]
        # dF2dJ = dF2[..., N:]
        
        # gf_loss = jnp.abs(jnp.gradient(jnp.gradient(dF2dJ, axis=-2), axis=-2)).sum() + jnp.abs(dF2dq - p).sum()
        # gf_loss = jnp.abs(dF2dq - p).sum()

        loss = const_loss + spread_loss * 0.3

        return loss, J

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, J), grads = grad_fn(model, generating_function, batch)
    metrics.update(loss=loss)
    optimizer.update(grads)


@nnx.jit
def gf_train_step(
    model: GeneratingFunction,
    motion_constant: MotionConstant,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch,
):
    def loss_fn(
        generating_function: GeneratingFunction, motion_constant: MotionConstant, batch
    ):
        q, p = batch

        J = motion_constant(jnp.concat((q, p), axis=-1))

        dF2dq, dF2dJ = generating_function.time_derivative(q, J)
        
        omega = jnp.gradient(jnp.sin(dF2dJ), axis=-2)
        omega_loss = ((omega - jnp.cos(dF2dJ) * (-1/(omega_0 * np.pi))) ** 2).mean()
        # phi_loss = jnp.abs(jnp.gradient(omega, axis=-2)).sum()
        # phi_loss = jnp.pow(omega - omega.mean(axis=-2)[..., None, :], 2).sum()
        # omega_spread_loss = -jnp.std(omega, axis=0).sum()
        
        p_loss = ((dF2dq - p) ** 2).mean()
        # p_loss = jnp.abs(jnp.sin(dF2dq) - jnp.sin(p)).sum() + jnp.abs(jnp.cos(dF2dq) - jnp.cos(p)).sum()
        # q_spread_loss = -jnp.std(dF2dq, axis=-2).sum()

        loss = p_loss + omega_loss * 0.1

        return loss, (p_loss, omega_loss)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, meta), grads = grad_fn(model, motion_constant, batch)
    metrics.update(loss=loss)
    optimizer.update(grads)

    return meta

# %%
mc_model = MotionConstant(1, rngs=nnx.Rngs(0))
gf_model = GeneratingFunction(1, rngs=nnx.Rngs(1), epsilon=0.05)

learning_rate = 0.005
momentum = 0.9

mc_opt = nnx.Optimizer(mc_model, optax.adamw(learning_rate, momentum))
gf_opt = nnx.Optimizer(gf_model, optax.adamw(learning_rate, momentum))

mc_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

gf_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
# %%
train_gf_after = 5000

step_list = []
mc_metric_history = {"train_loss": []}
gf_metric_history = {"train_loss": []}

for step, batch in enumerate(train_ds.as_numpy_iterator()):
    mc_meta = mc_train_step(mc_model, gf_model, mc_opt, mc_metrics, batch)
    if step >= train_gf_after:
        gf_meta = gf_train_step(gf_model, mc_model, gf_opt, gf_metrics, batch)

    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
        for metric, value in mc_metrics.compute().items():
            mc_metric_history[f"train_{metric}"].append(value)
        mc_metrics.reset()

        for metric, value in gf_metrics.compute().items():
            gf_metric_history[f"train_{metric}"].append(value)
        gf_metrics.reset()

        step_list.append(step)

        clear_output(wait=True)
        # Plot loss and accuracy in subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title("MC Loss")
        ax2.set_title("GF Loss")
        # ax2.set_title('Accuracy')
        ax1.plot(step_list, mc_metric_history["train_loss"], label="mc loss")
        ax2.plot(step_list, gf_metric_history["train_loss"], label="gf loss")
        ax2.set_yscale('log')
        # ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
        # ax1.legend()
        # ax2.legend()
        plt.show()
  # %% [markdown]
# ### Training Results
# Plot 1: True J calculated using elliptic integrals versus predicted J from motion constant model
#
# Plot 2: p from SHO transform versus recovered p = dF2/dq from learned generating function
#
# Plot 3: q from SHO transform versus phi = Q = dF2/dJ from learned generating function
#
# Plot 4: Time derivative of plot 3, jumps due to modulus discontinuity are replaced with the *median* value
# %%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
for i in range(5):
    H = hamiltonian_fn(data["x"][i].T)[0]
    true_J = jnp.array(list(map(lambda H: J_transform(G, F, H), H)))
    # true_phi = jnp.array(phi_transform(data["x"][i].T[0], H.mean(), 3))
    kappa = compute_kappa(H.mean(), 3)
    true_omega = jnp.pi * kappa/ellipk(1/kappa)

    ax1.plot(true_J, c="grey", linestyle="--", label="True J")
    J = mc_model(jnp.concat((train_phi[i], train_J[i]), axis=-1))
    ax1.plot(J, label="Pred $J$")
    ax1.set_title("True vs Pred $J$")
    ax1.set_xlabel(r"$J$")
    ax1.set_ylabel(r"$t$")

    dF2dq, dF2dJ = gf_model.time_derivative(train_phi[i], J)

    ax2.plot(dF2dq, label="Pred $p_n$")
    ax2.plot(train_J[i], c="grey", linestyle="--", label="True $p$")
    # ax2.plot(J, c="grey", linestyle="--", label="True $p$")
    ax2.set_title("True vs Pred $p$")

    ax3.plot(dF2dJ, label=r'$\phi$')
    ax3.plot(train_phi[i], c="grey", linestyle="--", label='$p$')
    # ax3.plot(jnp.sin(train_phi[i]), label='sin')
    # ax3.plot(jnp.gradient(jnp.sin(train_phi[i]), axis=-2), label='d/dt sin')
    # ax3.plot(jnp.cos(train_phi[i]), label='cos')
    ax3.set_title(r"$Q =\phi$ and $q$")

    ax4.plot(grad_ignore_jump(dF2dJ[:, 0]), label=r'$\phi$')
    ax4.plot(grad_ignore_jump(train_phi[i][:, 0]), c="grey", linestyle="--", label='$p$')
    ax4.set_title(r"$\dot Q = \dot \phi$ and $\dot q$")

# remove duplicate legend entries
ax1.legend(*[*zip(*{l:h for h,l in zip(*ax1.get_legend_handles_labels())}.items())][::-1])
ax2.legend(*[*zip(*{l:h for h,l in zip(*ax2.get_legend_handles_labels())}.items())][::-1])
ax3.legend(*[*zip(*{l:h for h,l in zip(*ax3.get_legend_handles_labels())}.items())][::-1])
# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

for i in range(100):
    x = data["x"][i][:, 0] + 1j * data["x"][i][:, 1]
    ax1.plot(jnp.real(x), jnp.imag(x))

    # print(jnp.abs(x).mean(), mc_model(data['x'][i]).mean())
    true_J = jnp.array(list(map(lambda H: J_transform(G, F, H), hamiltonian_fn(data["x"][i].T)[0])))
    pred_J = mc_model(jnp.concat((train_phi[i], train_J[i]), axis=-1))
    ax2.errorbar(true_J.mean(), pred_J.mean(), xerr=true_J.std(), yerr= pred_J.std(), c='tab:blue', marker='.')
    # ax2.errorbar(true_J.mean(), pred_J.mean()**1.5 * 1.2 + 3.4, xerr=true_J.std(), yerr= pred_J.std(), c='tab:blue', marker='.')
# ax2.plot([3.2,4.2], [3.2,4.2])
ax2.set_xlabel("True J")
ax2.set_ylabel("Pred J")
# %% [markdown]
# ### Find equations for motion constant predictor and generating function with sympy
# %%
default_pysr_params = dict(
	populations=40,
	procs=4,
	model_selection="best",
    output_directory='pysr_output',
    turbo=True,
)
# %%
mc_in = np.concatenate((train_phi, train_J), axis=-1)
mc_in = mc_in.reshape(-1, mc_in.shape[-1])
mc_out = [mc_model(jnp.concat((train_phi[i], train_J[i]), axis=-1)) for i in range(len(train_phi))]
mc_out = np.array(mc_out)[..., 0].flatten()

gf_in = np.concatenate((train_phi.reshape(-1, train_phi.shape[-1]), mc_out[..., None]), axis=-1)
gf_out = [gf_model(train_phi[i], mc_model(jnp.concat((train_phi[i], train_J[i]), axis=-1))) for i in range(len(train_phi))]
gf_out = np.array(gf_out)[..., 0].flatten()

selection = np.random.randint(0, mc_in.shape[0], 2_000)
mc_in = mc_in[selection]
mc_out = mc_out[selection]
gf_in = gf_in[selection]
gf_out = gf_out[selection]
# %%
mc_model_pysr = PySRRegressor(
	binary_operators=["+", "*", "-", "/", "^"],
	unary_operators=["sin", "exp", "tan", "asin"],
    constraints={"^": (1, 9)},
    nested_constraints={
        "sin": {"sin": 1, "tan": 1, "asin": 0, "^": 0},
        "tan": {"sin": 1, "tan": 1, "asin": 0, "^": 0},
        "asin": {"sin": 0, "tan": 0, "asin": 0, "^": 0},
        "^": {"sin": 1, "tan": 0, "asin": 0}},
    maxsize=40,
    batching=True,
	**default_pysr_params
)
# %%
mc_model_pysr.fit(mc_in, mc_out)
# %%
# mc_model_pysr = mc_model_pysr.from_file(run_directory='pysr_output/20250603_050349_qela2o')
# %%
print(mc_model_pysr.latex())
mc_model_pysr.sympy()
# %%
gf_model_pysr = PySRRegressor(
	binary_operators=["+", "*", "-", "/", "^"],
	unary_operators=["sin", "exp", "tan", "asin"],
    constraints={"^": (1, 9)},
    nested_constraints={
        "sin": {"sin": 1, "tan": 1, "asin": 0, "^": 0},
        "tan": {"sin": 1, "tan": 1, "asin": 0, "^": 0},
        "asin": {"sin": 0, "tan": 0, "asin": 0, "^": 0},
        "^": {"sin": 1, "tan": 0, "asin": 0}},
    maxsize=40,
    batching=True,
	**default_pysr_params
)
# %%
gf_model_pysr.fit(gf_in, gf_out)
# %%
# gf_model_pysr = gf_model_pysr.from_file(run_directory='pysr_output/20250603_062922_GCxKMN')
# %%
gf_model_pysr_jax = lambda x: gf_model_pysr.jax()['callable'](x, gf_model_pysr.jax()['parameters'])
print(gf_model_pysr.latex())
gf_model_pysr.sympy()
# %%
fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))

for i in range(100):
    x = data["x"][i][:, 0] + 1j * data["x"][i][:, 1]

    # print(jnp.abs(x).mean(), mc_model(data['x'][i]).mean())
    true_J = jnp.array(list(map(lambda H: J_transform(G, F, H), hamiltonian_fn(data["x"][i].T)[0])))
    pred_J = mc_model(jnp.concat((train_phi[i], train_J[i]), axis=-1))
    pred_J_pysr = mc_model_pysr.predict(jnp.concat((train_phi[i], train_J[i]), axis=-1))
    ax1.errorbar(true_J.mean(), pred_J.mean(), xerr=true_J.std(), yerr= pred_J.std(), c='tab:blue', marker='.')
    ax1.errorbar(true_J.mean(), pred_J_pysr.mean(), xerr=true_J.std(), yerr= pred_J_pysr.std(), c='tab:orange', marker='.')
ax1.set_xlabel("True J")
ax1.set_ylabel("Pred J")
# %%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
for i in range(5):
    H = hamiltonian_fn(data["x"][i].T)[0]
    true_J = jnp.array(list(map(lambda H: J_transform(G, F, H), H)))
    # true_phi = jnp.array(phi_transform(data["x"][i].T[0], H.mean(), 3))
    kappa = compute_kappa(H.mean(), 3)
    true_omega = jnp.pi * kappa/ellipk(1/kappa)

    ax1.plot(true_J, c="grey", linestyle="--", label="True J")
    J = mc_model(jnp.concat((train_phi[i], train_J[i]), axis=-1))
    J_pysr = mc_model_pysr.predict(jnp.concat((train_phi[i], train_J[i]), axis=-1))
    ax1.plot(J, c='black', label="Pred $J$")
    ax1.plot(J_pysr, label="PySR Pred $J$")
    ax1.set_title("True vs Pred $J$")
    ax1.set_xlabel(r"$J$")
    ax1.set_ylabel(r"$t$")

    dF2dq, dF2dJ = gf_model.time_derivative(train_phi[i], J)
    dF2dq_pysr, dF2dJ_pysr = nnx.grad(lambda q, J: gf_model_pysr_jax(jnp.concat((q, J), axis=-1)).sum(), argnums=[0,1])(train_phi[i], J_pysr[..., None])

    ax2.plot(train_J[i], c="grey", linestyle="--", label="True $p$")
    ax2.plot(dF2dq, c='black', label="Pred $p_n$")
    ax2.plot(dF2dq_pysr, label="PySR Pred $p_n$")
    # ax2.plot(J, c="grey", linestyle="--", label="True $p$")
    ax2.set_title("True vs Pred $p$")

    ax3.plot(train_phi[i], c="grey", linestyle="--", label='$p$')
    ax3.plot(dF2dJ, c='black', label=r'$\phi$')
    ax3.plot(dF2dJ_pysr, label=r'$\phi$')
    # ax3.plot(jnp.sin(train_phi[i]), label='sin')
    # ax3.plot(jnp.gradient(jnp.sin(train_phi[i]), axis=-2), label='d/dt sin')
    # ax3.plot(jnp.cos(train_phi[i]), label='cos')
    ax3.set_title(r"$Q =\phi$ and $q$")

    ax4.plot(grad_ignore_jump(train_phi[i][:, 0]), c="grey", linestyle="--", label='$p$')
    ax4.plot(grad_ignore_jump(dF2dJ[:, 0]), c='black', label=r'$\phi$')
    ax4.plot(grad_ignore_jump(dF2dJ_pysr[:, 0]), label=r'PySR $\phi$')
    ax4.set_title(r"$\dot Q = \dot \phi$ and $\dot q$")

# remove duplicate legend entries
ax1.legend(*[*zip(*{l:h for h,l in zip(*ax1.get_legend_handles_labels())}.items())][::-1])
ax2.legend(*[*zip(*{l:h for h,l in zip(*ax2.get_legend_handles_labels())}.items())][::-1])
ax3.legend(*[*zip(*{l:h for h,l in zip(*ax3.get_legend_handles_labels())}.items())][::-1])
# %%
import plotly.graph_objects as go
# %%
npts = 50
x = np.outer(np.linspace(data["x"].min(), data["x"].max(), npts), np.ones(npts))
y = x.copy().T
xp = x.reshape(-1)
yp = y.reshape(-1)

coord = x + 1j * y
nx = np.angle(coord)
ny = np.abs(coord)
nxp = nx.reshape(-1)
nyp = ny.reshape(-1)
# %%
H = hamiltonian_fn(np.stack((xp, yp)))[0]
true_J = np.array(list(map(lambda H: J_transform(G, F, H), H)))

J = mc_model(jnp.stack((nxp, nyp)).T)
J_pysr = mc_model_pysr.predict(jnp.stack((nxp, nyp)).T)
# %%
fig = go.Figure(data=[
    go.Surface(x=nx, y=ny, z=true_J.reshape(x.shape), opacity=0.9),
    # go.Surface(x=x, y=y, z=J.reshape(x.shape), opacity=0.8)
    go.Surface(x=nx, y=ny, z=J.reshape(x.shape)**1.5 * 1.2 + 3.4, opacity=0.9)
])
fig.update_layout(
    title={'text': 'J: True vs Learned Tranform'},
    scene={
        'xaxis': {'title': {'text': 'q'}},
        'yaxis': {'title': {'text': 'p'}},
        'zaxis': {'title': {'text': 'J'}}
    }
)
fig.show()
# %%
fig = go.Figure(data=[
    go.Surface(x=nx, y=ny, z=J.reshape(x.shape), opacity=0.8),
    go.Surface(x=nx, y=ny, z=J_pysr.reshape(x.shape), opacity=0.8)
])
fig.update_layout(
    title={'text': 'J: Learned vs PySR Tranform'},
    scene={
        'xaxis': {'title': {'text': 'q'}},
        'yaxis': {'title': {'text': 'p'}},
        'zaxis': {'title': {'text': 'J'}}
    }
)
fig.show()
# %%
