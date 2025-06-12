# %%
import numpy as np
import matplotlib.pyplot as plt

from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
# %%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
# %%
t = np.linspace(0, 5, 500)

xi_1 = 3 * np.exp(1j * t * 10) * np.exp(np.random.normal(0, 0.05, t.shape))
xi_2 = 2 * np.exp(1j * t * 4) * np.exp(np.random.normal(0, 0.05, t.shape))

c = 0.5 * np.array([[1, -1], [1, 1]])

x_1 = c[0, 0] * xi_1 + c[0, 1] * xi_2
x_2 = c[1, 0] * xi_1 + c[1, 1] * xi_2
X = np.stack((x_1, x_2))

c = c * np.array([3, 2])

fig, axs = plt.subplots(1, 2)
axs[0].plot(np.real(X[0]), np.imag(X[0]))
axs[0].set_aspect('equal')
axs[1].plot(np.real(X[1]), np.imag(X[1]))
axs[1].set_aspect('equal')
# %%
xs = ('x_1', 'x_2')
fmft_result = {}
for i,x in enumerate(xs):
    fmft_result[x] = fmft(t, X[i], 2)

print(fmft_result)
# %%
g_vec = np.zeros(len(xs))
g_vec = np.array(list(fmft_result['x_1'].keys()))
print(g_vec)
# %%
rotation_matrix = np.zeros((len(xs), len(xs)))

mode_angle = np.zeros(len(xs))
for i,x in enumerate(xs):
    freqs = np.array(list(fmft_result[x].keys()))
    for j, s in enumerate(g_vec):
        found_s = find_nearest(freqs, s)
        if np.abs((found_s - s)/s) > 0.1:
            continue
        rotation_matrix[i][j] = np.abs(fmft_result[x][found_s])

        if mode_angle[i] == 0:
            mode_angle[i] = np.angle(fmft_result[x][found_s])
        if mode_angle[i] != 0 and np.abs(mode_angle[i] - np.angle(fmft_result[x][found_s])) > np.pi/2:
            rotation_matrix[i][j] *= -1

print('true matrix')
print(c)
print()

print('un-normalized')
print(rotation_matrix)
print()

# normalize_cols = 2 / np.sum(np.abs(rotation_matrix), axis=0)
normalize_cols = 1/np.linalg.norm(rotation_matrix, axis=0)
rotation_matrix = rotation_matrix * normalize_cols

print('normalized')
print(rotation_matrix)

print(np.linalg.det(rotation_matrix))
# %%
Y = (X.T @ rotation_matrix.T).T
fig, axs = plt.subplots(1, 2)
axs[0].plot(np.real(Y[0]), np.imag(Y[0]))
axs[0].set_aspect('equal')
axs[1].plot(np.real(Y[1]), np.imag(Y[1]))
axs[1].set_aspect('equal')
# %%
