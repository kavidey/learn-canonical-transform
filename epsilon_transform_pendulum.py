# %%
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
data = get_dataset(samples=1024, seed=seed, noise_std=0.005)

N = 1
# %%
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

train_steps = 12000
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

    #     self.linear1 = nnx.Linear(2*N, 10, rngs=rngs, kernel_init=orthogonal_initializer)
    #     self.linear2 = nnx.Linear(10, 15, rngs=rngs, kernel_init=orthogonal_initializer)
    #     self.linear3 = nnx.Linear(15, 5, rngs=rngs, kernel_init=orthogonal_initializer)
    #     self.linear4 = nnx.Linear(5, 1, rngs=rngs, kernel_init=orthogonal_initializer)

    def __call__(self, x):
        # x = nnx.tanh(self.linear1(x))
        # x = nnx.tanh(self.linear2(x))
        # x = nnx.tanh(self.linear3(x))
        # x = self.linear4(x)
        return self.mlp(x)


class GeneratingFunction(nnx.Module):

    def __init__(self, N: int, rngs: nnx.Rngs):
        self.mlp = MLP(
            [2, 10, 15, 5, 1],
            initializer=orthogonal_initializer,
            activation=nnx.gelu,
            rngs=rngs,
        )
        # self.linear1 = nnx.Linear(2*N, 10, rngs=rngs, kernel_init=orthogonal_initializer)
        # self.linear2 = nnx.Linear(10, 15, rngs=rngs, kernel_init=orthogonal_initializer)
        # self.linear3 = nnx.Linear(15, 5, rngs=rngs, kernel_init=orthogonal_initializer)
        # self.linear4 = nnx.Linear(5, 1, rngs=rngs, kernel_init=orthogonal_initializer)

    def __call__(self, x):
        # x = nnx.tanh(self.linear1(x))
        # x = nnx.tanh(self.linear2(x))
        # x = nnx.tanh(self.linear3(x))
        # x = self.linear4(x)
        return self.mlp(x)
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

        # loss = jnp.abs(J - J_true).sum() + jnp.abs(jnp.gradient(J, axis=-2)).sum()

        # dF2 = nnx.grad(lambda x: generating_function(x).sum())(jnp.concat((q, J), axis=-1))
        # dF2dq = dF2[..., :N]
        # dF2dJ = dF2[..., N:]

        const_loss = jnp.pow(jnp.gradient(J, axis=-2), 2).sum()
        # const_loss = jnp.pow(J - J.mean(axis=-2)[..., None, :], 2).sum()
        spread_loss = -jnp.std(jnp.mean(J, axis=-2), axis=0).sum()
        # # gf_loss = jnp.abs(jnp.gradient(jnp.gradient(dF2dJ, axis=-2), axis=-2)).sum() + jnp.abs(dF2dq - p).sum()
        # gf_loss = jnp.abs(dF2dq - p).sum()

        # loss = const_loss + gf_loss
        # loss = gf_loss
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

        dF2 = nnx.grad(lambda x: generating_function(x).sum())(
            jnp.concat((q, J), axis=-1)
        )
        dF2dq = dF2[..., :N]
        dF2dJ = dF2[..., N:]

        # fix to get rid of jumps in the derivative caused by angle modulo
        # phi_est = dF2dJ
        # omega_median = jnp.median(phi_est, axis=-2)
        # omega_est = jnp.place(phi_est, jnp.abs(phi_est) > 2.85, phi_median.repeat(30, axis=-1)[..., None], inplace=False)
        
        # phi_loss = jnp.abs(jnp.gradient(jnp.gradient(dF2dJ, axis=-2), axis=-2)).sum()
        # phi_loss = 
        p_loss = jnp.abs(dF2dq - p).sum()

        loss = p_loss

        return loss, dF2dq

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, dF2dq), grads = grad_fn(model, motion_constant, batch)
    metrics.update(loss=loss)
    optimizer.update(grads)


# %%
mc_model = MotionConstant(1, rngs=nnx.Rngs(0))
gf_model = GeneratingFunction(1, rngs=nnx.Rngs(1))

learning_rate = 0.005
momentum = 0.9

mc_opt = nnx.Optimizer(mc_model, optax.adamw(learning_rate, momentum))
gf_opt = nnx.Optimizer(gf_model, optax.adamw(learning_rate, momentum))

mc_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

gf_metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
# %%
mc_metric_history = {"train_loss": []}
gf_metric_history = {"train_loss": []}

for step, batch in enumerate(train_ds.as_numpy_iterator()):
    mc_train_step(mc_model, gf_model, mc_opt, mc_metrics, batch)
    # gf_train_step(gf_model, mc_model, gf_opt, gf_metrics, batch)

    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
        for metric, value in mc_metrics.compute().items():
            mc_metric_history[f"train_{metric}"].append(value)
        mc_metrics.reset()

        for metric, value in gf_metrics.compute().items():
            gf_metric_history[f"train_{metric}"].append(value)
        gf_metrics.reset()

        clear_output(wait=True)
        # Plot loss and accuracy in subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title("MC Loss")
        ax2.set_title("GF Loss")
        # ax2.set_title('Accuracy')
        ax1.plot(mc_metric_history["train_loss"], label="mc loss")
        ax2.plot(gf_metric_history["train_loss"], label="gf loss")
        # ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
        # ax1.legend()
        # ax2.legend()
        plt.show()
# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
for i in range(5):
    true_J = jnp.array(list(map(lambda H: J_transform(2, 3, H), hamiltonian_fn(data["x"][i].T)[0])))
    true_phi = jnp.array(phi_transform(data["x"][i].T[0], hamiltonian_fn(data["x"][i].T).mean(), 3))

    J = mc_model(jnp.concat((train_phi[i], train_J[i]), axis=-1))
    ax1.plot(J)
    ax1.plot(true_J, c="grey", linestyle="--")
    ax1.set_title("J")

    dF2 = nnx.grad(lambda x: gf_model(x).sum())(
        jnp.concat((train_phi[i], J), axis=-1)
    )
    dF2dq = dF2[..., :N]
    dF2dJ = dF2[..., N:]

    ax2.plot(dF2dq)
    ax2.plot(train_phi[i], c="grey", linestyle="--")
    ax2.set_title("q")

    # ax3.plot(dF2dJ)
    ax3.plot(true_phi, c="grey", linestyle="--")
    ax3.set_title(r"$\phi$")
# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

for i in range(100):
    x = data["x"][i][:, 0] + 1j * data["x"][i][:, 1]
    ax1.plot(jnp.real(x), jnp.imag(x))

    # print(jnp.abs(x).mean(), mc_model(data['x'][i]).mean())
    true_J = jnp.array(list(map(lambda H: J_transform(2, 3, H), hamiltonian_fn(data["x"][i].T)[0])))
    pred_J = mc_model(data["x"][i])
    ax2.errorbar(true_J.mean(), pred_J.mean(), xerr=true_J.std(), yerr= pred_J.std(), c='tab:blue', marker='.')
ax2.set_xlabel("True J")
ax2.set_ylabel("Pred J")
# %%
