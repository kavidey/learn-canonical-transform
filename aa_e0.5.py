# %%
from pathlib import Path
import action_angle_tools
import numpy as np
import matplotlib.pyplot as plt
# %%
e05_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_e0.5' / 'npy'
e07_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_e0.7' / 'npy'
e09_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_e0.9' / 'npy'
e10_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_e_orig' / 'npy'
# %%
e05_sim, masses, rb_sim = action_angle_tools.load_sim(e05_dataset/"solarsystem_m1200.hires.npz", np.load(e05_dataset/"solarsystem_m1200.hires.npz", allow_pickle=True)['arr_0'][()]) 
psi = np.concat([e05_sim['x'], e05_sim['y']])

for i in range(1, len(rb_sim.particles)):
    p = rb_sim.particles[i]
    p.e = p.e * 0.5
# %%
axs = action_angle_tools.radial_plots(psi[:action_angle_tools.N], psi[action_angle_tools.N:], pl_list=action_angle_tools.planets, sym_axes=
                                False)
# %%
psi_decoupled, ecc_rotation_matrix_opt_T, inc_rotation_matrix_opt_T = action_angle_tools.decouple_modes(psi[:action_angle_tools.N], psi[action_angle_tools.N:], rb_sim, masses)
psi_decoupled = np.concat(psi_decoupled)
# %%
axs = action_angle_tools.radial_plots(psi, psi_decoupled, sym_axes=False)
# %%
planet_fmft = action_angle_tools.get_planet_fmft(action_angle_tools.psi_planet_list, e05_sim['time'], psi_decoupled, N=14, display=True)
# TODO: how to make sure that the omegas are in the right order
# %%
g_vec = np.zeros(8)
s_vec = np.zeros(8)

for i in range(len(action_angle_tools.planets)):
    freqs = list(planet_fmft[action_angle_tools.psi_planet_list[:action_angle_tools.N][i]].keys())
    for f in freqs:
        if (np.abs((g_vec[:i] / f) - 1) < 1e-4).any():
            continue
        g_vec[i] = f
        break

for i in range(len(action_angle_tools.planets)):
    freqs = list(planet_fmft[action_angle_tools.psi_planet_list[action_angle_tools.N:][i]].keys())
    for f in freqs:
        if i != np.argmax(masses) and np.abs(f*action_angle_tools.TO_ARCSEC_PER_YEAR) < 0.001:
            continue
        if (np.abs((s_vec[:i] / f) - 1) < 1e-4).any():
            continue
        s_vec[i] = f
        break

g_amp = np.zeros(8, dtype=np.complex128)
s_amp = np.zeros(8, dtype=np.complex128)

g_amp[0] = planet_fmft['Mercury_X'][g_vec[0]]
g_amp[1] = planet_fmft['Venus_X'][g_vec[1]]
g_amp[2] = planet_fmft['Earth_X'][g_vec[2]]
g_amp[3] = planet_fmft['Mars_X'][g_vec[3]]
g_amp[4] = planet_fmft['Jupiter_X'][g_vec[4]]
g_amp[5] = planet_fmft['Saturn_X'][g_vec[5]]
g_amp[6] = planet_fmft['Uranus_X'][g_vec[6]]
g_amp[7] = planet_fmft['Neptune_X'][g_vec[7]]

s_amp[0] = planet_fmft['Mercury_Y'][s_vec[0]]
s_amp[1] = planet_fmft['Venus_Y'][s_vec[1]]
s_amp[2] = planet_fmft['Earth_Y'][s_vec[2]]
s_amp[3] = planet_fmft['Mars_Y'][s_vec[3]]
s_amp[4] = planet_fmft['Jupiter_Y'][s_vec[4]]
s_amp[5] = planet_fmft['Saturn_Y'][s_vec[5]]
s_amp[6] = planet_fmft['Uranus_Y'][s_vec[6]]
s_amp[7] = planet_fmft['Neptune_Y'][s_vec[7]]

omega_vec = np.concat([g_vec, s_vec])
omega_amp = np.concat([g_amp, s_amp])

print(action_angle_tools.true_omega_vec)
print((omega_vec * action_angle_tools.TO_ARCSEC_PER_YEAR).round(3))
print(omega_amp)
# %%
psi_cancelled, trans_fns = action_angle_tools.cancel_frequencies(
    psi_decoupled, e05_sim['time'],
    omega_vec, omega_amp,
    iterations=[3], debug=True
)
# %%
axs = action_angle_tools.radial_plots(psi_decoupled, psi_cancelled)
# %%
plt.plot(np.abs(psi[0]), label="orig")
plt.plot(np.abs(psi_decoupled[0]), label="decoupled")
plt.plot(np.abs(psi_cancelled[0]), label="cancelled")
plt.legend()
# %%
X, C_opts, found_combs = action_angle_tools.pca_combs(psi_decoupled)

gamma_0 = np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
C_ecc = gamma_0 @ X

gamma_1 = np.array([0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0])
C_inc = gamma_1 @ X

gamma_opt = found_combs[-1]
C_opt = C_opts[-1]
# %%
action_angle_tools.plot_action(psi[0], e05_sim['time'], take_mag=True, label=r"$\Psi$")
action_angle_tools.plot_action(psi_decoupled[0], e05_sim['time'], take_mag=True, label=r"$\Psi_{decoupled}$")
action_angle_tools.plot_action(psi_cancelled[0], e05_sim['time'], take_mag=True, label=r"$\Psi_{cancelled}$")
action_angle_tools.plot_action(C_ecc, e05_sim['time'], take_mag=False, label="$C_{ecc}$")
action_angle_tools.plot_action(C_inc , e05_sim['time'], take_mag=False, label="$C_{inc}$")
action_angle_tools.plot_action(C_opt , e05_sim['time'], take_mag=False, label="$C_{opt}$")
plt.legend()
# %%
e05_sim_long, masses, rb_sim = action_angle_tools.load_sim(e05_dataset/"solarsystem_m1200.npz", np.load(e05_dataset/"solarsystem_m1200.npz", allow_pickle=True)['arr_0'][()]) 
psi_long = np.concat([e05_sim_long['x'], e05_sim_long['y']])
# %%
psi_long_decoupled = np.concat(((np.linalg.inv(ecc_rotation_matrix_opt_T) @ psi_long[:action_angle_tools.N]), (np.linalg.inv(inc_rotation_matrix_opt_T) @ psi_long[action_angle_tools.N:])))
psi_long_cancelled = action_angle_tools.apply_sequential_transforms(psi_long_decoupled, trans_fns)

X_long = np.real(psi_long_decoupled * np.conj(psi_long_decoupled))
C_ecc_long = gamma_0 @ X_long
C_inc_long = gamma_1 @ X_long
C_opt_long = gamma_opt @ X_long
# %%
action_angle_tools.plot_action(psi_long[0], e05_sim_long['time'], take_mag=True, label=r"$\Psi$")
action_angle_tools.plot_action(psi_long_decoupled[0], e05_sim_long['time'], take_mag=True, label=r"$\Psi_{decoupled}$")
action_angle_tools.plot_action(psi_long_cancelled[0], e05_sim_long['time'], take_mag=True, label=r"$\Psi_{cancelled}$")
action_angle_tools.plot_action(C_ecc_long, e05_sim_long['time'], take_mag=False, label="$C_{ecc}$")
action_angle_tools.plot_action(C_inc_long, e05_sim_long['time'], take_mag=False, label="$C_{inc}$")
action_angle_tools.plot_action(C_opt_long, e05_sim_long['time'], take_mag=False, label="$C_{opt}$")
plt.legend()
# %%
