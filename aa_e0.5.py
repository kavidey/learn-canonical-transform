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

# swap venus and jupiter
psi_decoupled[[1,4]] = psi_decoupled[[4,1]]
psi_decoupled[[10,13,11]] = psi_decoupled[[13,11,10]]
psi_decoupled[[12,14,15]] = psi_decoupled[[15,12,14]]
# %%
axs = action_angle_tools.radial_plots(psi, psi_decoupled, sym_axes=False)
# %%
planet_fmft = action_angle_tools.get_planet_fmft(action_angle_tools.psi_planet_list, e05_sim['time'], psi_decoupled, N=14, display=True)
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
psi_cancelled, trans_fns, combs = action_angle_tools.cancel_frequencies(
    psi_decoupled, e05_sim['time'],
    omega_vec, omega_amp,
    iterations=[1,3,5], omega_pct_thresh=2e-5,
    debug=True
)
# %%
axs = action_angle_tools.radial_plots(psi_decoupled, psi_cancelled)
# %%
plt_lim = 100
planet_fmft = action_angle_tools.get_planet_fmft(action_angle_tools.psi_planet_list, e05_sim['time'], psi_decoupled, N=14, fmft_alg="default", display=True)
# planet_fmft_cancelled = action_angle_tools.get_planet_fmft(action_angle_tools.psi_planet_list, e05_sim['time'], psi_cancelled, N=14, fmft_alg="default", display=True)
fig, axs = plt.subplots(2, action_angle_tools.N, figsize=(10,3), sharex=True)
for i, pl in enumerate(action_angle_tools.planets):
    axs[0][i].set_title(pl)

    fmft_recon_x = np.sum([amp * np.exp(1j*freq*e05_sim['time']) for freq,amp in planet_fmft[action_angle_tools.psi_planet_list[i]].items()],axis=0)
    fmft_recon_y = np.sum([amp * np.exp(1j*freq*e05_sim['time']) for freq,amp in planet_fmft[action_angle_tools.psi_planet_list[i+action_angle_tools.N]].items()],axis=0)
    with plt.rc_context({"lines.linewidth":0.2}):
        axs[0][i].plot(e05_sim['time'][plt_lim:]*action_angle_tools.TO_YEAR*1e-6, fmft_recon_x[plt_lim:] * np.conj(fmft_recon_x)[plt_lim:], alpha=0.5, c='grey', label='FMFT')
        axs[1][i].plot(e05_sim['time'][plt_lim:]*action_angle_tools.TO_YEAR*1e-6, fmft_recon_y[plt_lim:] * np.conj(fmft_recon_y)[plt_lim:], alpha=0.5, c='grey', label='FMFT')

        # fmft_recon_x = np.sum([amp * np.exp(1j*freq*e05_sim['time']) for freq,amp in planet_fmft_cancelled[action_angle_tools.psi_planet_list[i]].items()],axis=0)
        # fmft_recon_y = np.sum([amp * np.exp(1j*freq*e05_sim['time']) for freq,amp in planet_fmft_cancelled[action_angle_tools.psi_planet_list[i+action_angle_tools.N]].items()],axis=0)
        # axs[0][i].plot(e05_sim['time'][100:], fmft_recon_x[100:] * np.conj(fmft_recon_x)[100:], label='FMFT Recon cancelled', c='black')
        # axs[1][i].plot(e05_sim['time'][100:], fmft_recon_y[100:] * np.conj(fmft_recon_y)[100:], c='black')

        axs[0][i].plot(e05_sim['time'][plt_lim:]*action_angle_tools.TO_YEAR*1e-6, psi_decoupled[i][plt_lim:] * np.conj(psi_decoupled[i])[plt_lim:], label='Decoup.')
        axs[1][i].plot(e05_sim['time'][plt_lim:]*action_angle_tools.TO_YEAR*1e-6, psi_decoupled[i+action_angle_tools.N][plt_lim:] * np.conj(psi_decoupled[i+action_angle_tools.N])[plt_lim:], label='Decoup.')

        axs[0][i].plot(e05_sim['time'][plt_lim:]*action_angle_tools.TO_YEAR*1e-6, psi_cancelled[i][plt_lim:] * np.conj(psi_cancelled[i])[plt_lim:], alpha=0.8, label='Canc.')
        axs[1][i].plot(e05_sim['time'][plt_lim:]*action_angle_tools.TO_YEAR*1e-6, psi_cancelled[i+action_angle_tools.N][plt_lim:] * np.conj(psi_cancelled[i+action_angle_tools.N])[plt_lim:], alpha=0.8, label='Canc.')

    axs[0][i].set_ylim(-axs[0][i].get_ylim()[1] / 10, axs[0][i].get_ylim()[1] * 1.5)
    axs[1][i].set_ylim(-axs[1][i].get_ylim()[1] / 10, axs[1][i].get_ylim()[1] * 1.5)
axs[0][0].set_ylabel("Eccentricity")
axs[1][0].set_ylabel("Inclination")
axs[0][7].legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.text(0.5, -0.04, 'Myr', ha='center')

plt.tight_layout(w_pad=0.4, h_pad=1)
plt.savefig("figs/frequency-canceling-timedom-e05.pdf")
# %%
def calc_fft(time, x):
    fourier = np.fft.fft(x * np.hanning(time.shape[-1]))
    fourier = np.fft.fftshift(fourier)
    freq = np.fft.fftfreq(time.shape[-1], d=time[1]) * action_angle_tools.TO_ARCSEC_PER_YEAR * 2 * np.pi
    freq = np.fft.fftshift(freq)
    fourier_amp = np.abs(fourier)

    return freq, fourier_amp

fig, axs = plt.subplots(2, action_angle_tools.N, figsize=(10,2.5))
for i, pl in enumerate(action_angle_tools.planets):
    axs[0][i].set_title(pl)

    fmft_recon_x = np.sum([amp * np.exp(1j*freq*e05_sim['time']) for freq,amp in planet_fmft[action_angle_tools.psi_planet_list[i]].items()],axis=0)
    fmft_recon_y = np.sum([amp * np.exp(1j*freq*e05_sim['time']) for freq,amp in planet_fmft[action_angle_tools.psi_planet_list[i+action_angle_tools.N]].items()],axis=0)
    
    with plt.rc_context({"lines.linewidth":0.6}):
        axs[0][i].plot(*calc_fft(e05_sim['time'], fmft_recon_x), label='FMFT', alpha=0.5, c='grey')
        axs[1][i].plot(*calc_fft(e05_sim['time'], fmft_recon_y), alpha=0.5, c='grey')

        axs[0][i].plot(*calc_fft(e05_sim['time'], psi_decoupled[i]), label='Decoup.')
        axs[1][i].plot(*calc_fft(e05_sim['time'], psi_decoupled[i+action_angle_tools.N]))

        axs[0][i].plot(*calc_fft(e05_sim['time'], psi_cancelled[i]), label='Canc.', alpha=0.8)
        axs[1][i].plot(*calc_fft(e05_sim['time'], psi_cancelled[i+action_angle_tools.N]), alpha=0.8)

    axs[0][i].set_xlim(omega_vec[i] * action_angle_tools.TO_ARCSEC_PER_YEAR - 2, omega_vec[i] * action_angle_tools.TO_ARCSEC_PER_YEAR + 2)
    axs[1][i].set_xlim(omega_vec[i+action_angle_tools.N] * action_angle_tools.TO_ARCSEC_PER_YEAR - 2, omega_vec[i+action_angle_tools.N] * action_angle_tools.TO_ARCSEC_PER_YEAR + 2)
axs[0][0].set_ylabel("Eccentricity")
axs[1][0].set_ylabel("Inclination")
axs[0][7].legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.text(0.5, -0.04, 'Frequency $\omega$ ["/yr]', ha='center')

plt.tight_layout(w_pad=0.2, h_pad=1)
plt.savefig("figs/frequency-canceling-freqdom-e05.pdf")
# %%
short_time = e05_sim['time']*action_angle_tools.TO_YEAR*1e-6
plt.figure(figsize=(6,2.5))
psi_curr = psi_decoupled
plt.plot(short_time, psi_curr[10] * np.conj(psi_curr[10]), color="grey")
plt.plot(short_time, psi_curr[11] * np.conj(psi_curr[11]), color="grey")
plt.plot(short_time, psi_curr[10] * np.conj(psi_curr[10]) + psi_curr[11] * np.conj(psi_curr[11]), color="grey")

# psi_curr = psi_cancelled
# planet_fmft = action_angle_tools.get_planet_fmft(action_angle_tools.psi_planet_list, e05_sim['time'], psi_curr, N=14, fmft_alg="default", display=True)
# earth_y = np.sum([amp * np.exp(1j*freq*e05_sim['time']) for freq,amp in planet_fmft["Earth_Y"].items()],axis=0)
# mars_y = np.sum([amp * np.exp(1j*freq*e05_sim['time']) for freq,amp in planet_fmft["Mars_Y"].items()],axis=0)

earth_y = psi_cancelled[10]
mars_y = psi_cancelled[11]

plt.plot(short_time, earth_y * np.conj(earth_y), label=r"$\mathcal{Y}_3$")
plt.plot(short_time, mars_y * np.conj(mars_y), label=r"$\mathcal{Y}_4$")
plt.plot(short_time, earth_y * np.conj(earth_y) + mars_y * np.conj(mars_y), label=r"$\mathcal{Y}_3 + \mathcal{Y}_4$")

plt.xlim(0,30)
plt.ylabel("Action")
plt.xlabel("Myr")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig("figs/e05-action-combinations-debug.eps")
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
