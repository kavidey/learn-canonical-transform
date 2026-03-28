# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import simple_harmonic_oscillator as sho
# %%
def aa_plot(axs, J, phi, **kwargs):
	axs[0].set_aspect('equal')
	
	axs[0].plot(np.cos(phi)*J,  np.sin(phi)*J, **kwargs)
	axs[1].plot(phi, J, **kwargs)
# %%
dt = 0.1
omega0_val = 1
h_f, dhdx_bar_f, xp_f = sho.hamiltonian_system()
# %%
desired_h = 1.5
x0 = sho.solve_for_x(h_f, desired_h, omega0_val)

t_span = (0, 15)
t_eval = np.arange(*t_span, dt)
_, J, phi = sho.sample_trajectory(dhdx_bar_f, x0, omega0_val, t_span, t_eval)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
aa_plot(axs, J, phi)
# %%
X = np.concat(([J], [phi]))
X = X - X.mean(axis=-1)[..., None]
X /= np.max(X)
pca = PCA().fit(X.T / X.mean())

found_combs = []
for i in reversed(range(1, 6)):
    # gamma_n = pca.components_[-i] / np.median(np.abs(pca.components_[-i]))
    gamma_n = pca.components_[-i] / (np.max(np.abs(pca.components_[-i])) / 2)
    gamma_n = gamma_n.round()
    found_combs.append(gamma_n)
    C_opt = gamma_n @ X
    C_opt_hat = C_opt / np.linalg.norm(gamma_n)
    plt.plot(t_eval, C_opt_hat - C_opt_hat.mean(), label=r"$\hat C_{opt,"+str(i)+"}$")

# plt.xlim(left=t[100], right=t[-1])
plt.legend()
plt.xlabel("Myr")
plt.show()
# %%
