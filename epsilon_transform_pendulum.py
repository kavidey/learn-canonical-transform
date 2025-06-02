# %%
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

seed = 42
data_dir = Path('datasets')
# %%
from scipy.special import ellipk, ellipe, ellipkinc, ellipeinc

# Define 𝜅
def compute_kappa(E, F):
    return np.sqrt(0.5 * (1 + E/F))

# Define η
def compute_eta(kappa, phi):
    return np.arcsin(np.sin(0.5 * phi) / kappa)

# Define the transformation into action (J)
def J_transform(G, F, E):
    kappa = compute_kappa(E, F)
    R = np.sqrt(F/G)
    
    if kappa < 1:
        J = R * 8/np.pi * ellipe(kappa**2) - (1 - kappa**2) * ellipk(kappa**2)
    
    else:
        J = R * 8/np.pi * 1/2 * kappa * ellipe(kappa * (-2))

    return J

# Define the transformation into angle (phi)
def phi_transform(theta, E, F):
    kappa = compute_kappa(E, F)
    eta = compute_eta(kappa, theta)
    eta_real = np.real(eta)

    if kappa < 1:
        phi = np.pi/2 * (ellipk(kappa ** 2)) ** (-1) * ellipkinc(eta_real, kappa**2)
    
    else:
        phi = np.pi/2 * 2 * ellipk(kappa ** (-2)) ** (-1) * ellipkinc(0.5 * theta, kappa ** (-2))

    return phi
# %%
np.random.seed(seed)
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
    def __init__(self, N: int, rngs: nnx.Rngs):
        self.mlp = MLP(
            [2, 10, 15, 5, 1],
            initializer=orthogonal_initializer,
            activation=nnx.gelu,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.mlp(x)


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
        
        omega = jnp.gradient(dF2dJ, axis=-2)
        omega_loss = ((omega - jnp.cos(dF2dJ)) ** 2).mean()
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
        # ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
        # ax1.legend()
        # ax2.legend()
        plt.show()
# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
for i in range(5):
    H = hamiltonian_fn(data["x"][i].T)[0]
    true_J = jnp.array(list(map(lambda H: J_transform(2, 3, H), H)))
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
    # ax3.plot(jnp.gradient(dF2dJ.T[0]))
    # ax3.plot(true_phi)
    # ax3.plot(jnp.gradient(true_phi), c="grey", linestyle="--")
    # ax3.axhline(true_omega, c="grey", linestyle="--")
    ax3.plot(train_phi[i], c="grey", linestyle="--", label='$p$')
    ax3.set_title(r"$Q =\phi$ and $q$")

ax1.legend(*[*zip(*{l:h for h,l in zip(*ax1.get_legend_handles_labels())}.items())][::-1])
ax2.legend(*[*zip(*{l:h for h,l in zip(*ax2.get_legend_handles_labels())}.items())][::-1])
ax3.legend(*[*zip(*{l:h for h,l in zip(*ax3.get_legend_handles_labels())}.items())][::-1])
# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

for i in range(100):
    x = data["x"][i][:, 0] + 1j * data["x"][i][:, 1]
    ax1.plot(jnp.real(x), jnp.imag(x))

    # print(jnp.abs(x).mean(), mc_model(data['x'][i]).mean())
    true_J = jnp.array(list(map(lambda H: J_transform(2, 3, H), hamiltonian_fn(data["x"][i].T)[0])))
    pred_J = mc_model(jnp.concat((train_phi[i], train_J[i]), axis=-1))
    ax2.errorbar(true_J.mean(), pred_J.mean(), xerr=true_J.std(), yerr= pred_J.std(), c='tab:blue', marker='.')
ax2.set_xlabel("True J")
ax2.set_ylabel("Pred J")
# %%
