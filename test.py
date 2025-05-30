# %%
import jax
import jax.numpy as jnp
import jax.random as jnr
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
import tensorflow_datasets as tfds

from data.spring import get_dataset

from flax import nnx
import optax
import orbax.checkpoint as ocp

seed = 42
# %%
np.random.seed(seed)
data = get_dataset(samples=1024, seed=seed, noise_std=0.01)

N = 1

# %%
train_steps = 12000
eval_every = 200
batch_size = 64

train_x = data['x']
# test_x = data['test_x']

train_dxdt = data['dx']
# test_dxdt = data['test_dx']

train_ds = tf.data.Dataset.from_tensor_slices(train_x)
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

    def __init__(self, N: int, rngs: nnx.Rngs):
        self.mlp = MLP(
            [2, 10, 15, 5, 1],
            initializer=orthogonal_initializer,
            activation=nnx.gelu,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.mlp(x)
# %%
@nnx.jit
def mc_train_step(model: MotionConstant, generating_function: GeneratingFunction, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    def loss_fn(motion_constant: MotionConstant, generating_function: GeneratingFunction, batch):
        q = batch[..., :N]
        p = batch[..., N:]
    
        J = motion_constant(jnp.concat((q, p), axis=-1))

        dF2 = nnx.grad(lambda x: generating_function(x).sum())(jnp.concat((q, J), axis=-1))
        dF2dq = dF2[..., :N]
        dF2dJ = dF2[..., N:]

        const_loss = jnp.abs(jnp.gradient(J, axis=-2)).sum()
        # const_loss = jnp.abs(J - J.mean(axis=-2)[..., None, :]).sum()
        spread_loss = -jnp.std(jnp.mean(J, axis=-2), axis=0).sum()
        # gf_loss = jnp.abs(jnp.gradient(jnp.gradient(dF2dJ, axis=-2), axis=-2)).sum() + jnp.abs(dF2dq - p).sum()
        gf_loss = jnp.abs(dF2dq - p).sum()

        loss = const_loss + spread_loss * 10 #+ gf_loss
        # loss = gf_loss

        return loss, (J, const_loss, spread_loss)
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, meta), grads = grad_fn(model, generating_function, batch)
    metrics.update(loss=loss)
    optimizer.update(grads)

    return meta

@nnx.jit
def gf_train_step(model: GeneratingFunction, motion_constant: MotionConstant, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    def loss_fn(generating_function: GeneratingFunction, motion_constant: MotionConstant, batch):
        q = batch[..., :N]
        p = batch[..., N:]

        J = motion_constant(jnp.concat((q, p), axis=-1))

        dF2 = nnx.grad(lambda x: generating_function(x).sum())(jnp.concat((q, J), axis=-1))
        dF2dq = dF2[..., :N]
        dF2dJ = dF2[..., N:]

        # loss = jnp.abs(jnp.gradient(jnp.gradient(dF2dJ, axis=-2), axis=-2)).sum() + jnp.abs(dF2dq - p).sum()
        loss = jnp.abs(dF2dq - p).sum()

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

mc_metrics = nnx.MultiMetric(
    loss = nnx.metrics.Average('loss')
)

gf_metrics = nnx.MultiMetric(
    loss = nnx.metrics.Average('loss')
)
# %%
mc_metric_history = {
    'train_loss': []
}
gf_metric_history = {
    'train_loss': []
}

for step, batch in enumerate(train_ds.as_numpy_iterator()):
    mc_train_step(mc_model, gf_model, mc_opt, mc_metrics, batch)
    # gf_train_step(gf_model, mc_model, gf_opt, gf_metrics, batch)

    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
        for metric, value in mc_metrics.compute().items():
            mc_metric_history[f'train_{metric}'].append(value)
        mc_metrics.reset()

        for metric, value in gf_metrics.compute().items():
            gf_metric_history[f'train_{metric}'].append(value)
        gf_metrics.reset()

        clear_output(wait=True)
        # Plot loss and accuracy in subplots
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
        ax1.set_title('Loss')
        # ax2.set_title('Accuracy')
        ax1.plot(mc_metric_history['train_loss'], label='mc loss')
        ax1.plot(gf_metric_history['train_loss'], label='gf loss')
        # ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
        ax1.legend()
        # ax2.legend()
        plt.show()

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
for i in range(5):
    J = mc_model(data['x'][i])
    ax1.plot(J)
    ax1.set_title('J')

    dF2 = nnx.grad(lambda x: gf_model(x).sum())(jnp.concat((data['x'][i][..., :N], J), axis=-1))
    dF2dq = dF2[..., :N]
    dF2dJ = dF2[..., N:]

    ax2.plot(dF2dq)
    ax2.plot(data['x'][i][...,:N], c='grey', linestyle='--')
    ax2.set_title("q")

    ax3.plot(dF2dJ)
    ax3.set_title(r"$\phi$")
# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

true_J = []
pred_J = []
for i in range(100):
    x = data['x'][i][:, 0] + 1j * data['x'][i][:, 1]
    plt.plot(jnp.real(x), jnp.imag(x))

    # print(jnp.abs(x).mean(), mc_model(data['x'][i]).mean())
    true_J.append(jnp.abs(x).mean())
    pred_J.append(mc_model(data['x'][i]).mean())

ax2.scatter(true_J, pred_J)
ax2.set_xlabel("True J")
ax2.set_ylabel("Pred J")
# %%
# %%
