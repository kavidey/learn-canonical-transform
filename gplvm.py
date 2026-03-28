# %%
import numpy as np

from GPy.models import GPLVM
from GPy.kern import RBF

import matplotlib.pyplot as plt
# %%
def make_parabola_data(n=100, sigma2=1):
	# x1 = np.sqrt(np.linspace(0, 3, n//2))
	# x1 = np.concatenate([-x1, x1])
	x1 = np.linspace(-1.5, 1.5, n)
	x1 = np.sort(x1)
	noise = np.random.normal(scale=np.sqrt(sigma2), size=n)
	x2 = x1**2 + noise
	return np.vstack([x1, x2]).T, noise

n = 200
Y, noise = make_parabola_data(n=n, sigma2=0.1)

plt.scatter(*Y.T)
# %%
# gplvm = GPLVM(X, 1, init='PCA')
model = GPLVM(Y, 1, normalizer=True)
model.optimize(messages=True, max_f_eval=10000)
# %%
model.plot()
# %%
plt.scatter(*Y.T, c=model.X)
# plt.scatter(*gplvm.X.T)
# plt.plot(*gplvm.X.T)
# %%
