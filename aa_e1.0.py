# %%
from pathlib import Path
import action_angle_tools
import numpy as np
import matplotlib.pyplot as plt
# %%
e05_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_e0.5' / 'npy'
e07_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_e0.7' / 'npy'
e09_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_e0.9' / 'npy'
e10_dataset = Path('datasets') / 'e_factor' / 'wf512_jac_orig' / 'npy'
# %%
e10_sim, masses, rb_sim = action_angle_tools.load_sim(e10_dataset/"solarsystem_m560.hires.npz", np.load(e10_dataset/"solarsystem_m560.hires.npz", allow_pickle=True)['arr_0'][()]) 
psi = np.concat([e10_sim['x'], e10_sim['y']])
# %%
axs = action_angle_tools.radial_plots(psi[:action_angle_tools.N], psi[action_angle_tools.N:], pl_list=action_angle_tools.planets, sym_axes=
                                False)
# %%
psi_decoupled, ecc_rotation_matrix_opt_T, inc_rotation_matrix_opt_T = action_angle_tools.decouple_modes(psi[:action_angle_tools.N], psi[action_angle_tools.N:], rb_sim, masses)
psi_decoupled = np.concat(psi_decoupled)

psi_decoupled[[8,9,10,11,12,13,14,15]] = psi_decoupled[[9,12,13,10,15,11,8,14]]
# %%
# axs = action_angle_tools.radial_plots(psi, psi_decoupled, sym_axes=False)
# %%
planet_fmft = action_angle_tools.get_planet_fmft(action_angle_tools.psi_planet_list, e10_sim['time'], psi_decoupled, N=14, display=True)
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
    psi_decoupled, e10_sim['time'],
    omega_vec, omega_amp,
    iterations=[1,3,5], omega_pct_thresh=2e-5,
    min_freq=20.0/action_angle_tools.TO_ARCSEC_PER_YEAR,
    debug=True
)
# %%
axs = action_angle_tools.radial_plots(psi_decoupled, psi_cancelled)
# %%
plt_lim = 100
planet_fmft = action_angle_tools.get_planet_fmft(action_angle_tools.psi_planet_list, e10_sim['time'], psi_decoupled, N=14, fmft_alg="custom", display=True)
fig, axs = plt.subplots(2, action_angle_tools.N, figsize=(20,5))
for i, pl in enumerate(action_angle_tools.planets):
    axs[0][i].set_title(pl)

    fmft_recon_x = np.sum([amp * np.exp(1j*freq*e10_sim['time']) for freq,amp in planet_fmft[action_angle_tools.psi_planet_list[i]].items()],axis=0)
    fmft_recon_y = np.sum([amp * np.exp(1j*freq*e10_sim['time']) for freq,amp in planet_fmft[action_angle_tools.psi_planet_list[i+action_angle_tools.N]].items()],axis=0)
    axs[0][i].plot(e10_sim['time'][plt_lim:], fmft_recon_x[plt_lim:] * np.conj(fmft_recon_x)[plt_lim:], label='FMFT Recon', alpha=0.5, c='grey')
    axs[1][i].plot(e10_sim['time'][plt_lim:], fmft_recon_y[plt_lim:] * np.conj(fmft_recon_y)[plt_lim:], alpha=0.5, c='grey')

    axs[0][i].plot(e10_sim['time'][plt_lim:], psi_decoupled[i][plt_lim:] * np.conj(psi_decoupled[i])[plt_lim:], label='Decoupled')
    axs[1][i].plot(e10_sim['time'][plt_lim:], psi_decoupled[i+action_angle_tools.N][plt_lim:] * np.conj(psi_decoupled[i+action_angle_tools.N])[plt_lim:])

    axs[0][i].plot(e10_sim['time'][plt_lim:], psi_cancelled[i][plt_lim:] * np.conj(psi_cancelled[i])[plt_lim:], label='Cancelled', alpha=0.8)
    axs[1][i].plot(e10_sim['time'][plt_lim:], psi_cancelled[i+action_angle_tools.N][plt_lim:] * np.conj(psi_cancelled[i+action_angle_tools.N])[plt_lim:], alpha=0.8)

    axs[0][i].set_ylim(-axs[0][i].get_ylim()[1] / 10, axs[0][i].get_ylim()[1] * 1.5)
    axs[1][i].set_ylim(-axs[1][i].get_ylim()[1] / 10, axs[1][i].get_ylim()[1] * 1.5)
axs[0][0].set_ylabel("Eccentricity")
axs[1][0].set_ylabel("Inclination")
axs[0][0].legend()
plt.show()
# %%
def calc_fft(time, x):
    fourier = np.fft.fft(x * np.hanning(time.shape[-1]))
    fourier = np.fft.fftshift(fourier)
    freq = np.fft.fftfreq(time.shape[-1], d=time[1]) * action_angle_tools.TO_ARCSEC_PER_YEAR * 2 * np.pi
    freq = np.fft.fftshift(freq)
    fourier_amp = np.abs(fourier)

    return freq, fourier_amp

fig, axs = plt.subplots(2, action_angle_tools.N, figsize=(20,5))
for i, pl in enumerate(action_angle_tools.planets):
    axs[0][i].set_title(pl)

    fmft_recon_x = np.sum([amp * np.exp(1j*freq*e10_sim['time']) for freq,amp in planet_fmft[action_angle_tools.psi_planet_list[i]].items()],axis=0)
    fmft_recon_y = np.sum([amp * np.exp(1j*freq*e10_sim['time']) for freq,amp in planet_fmft[action_angle_tools.psi_planet_list[i+action_angle_tools.N]].items()],axis=0)
    
    axs[0][i].plot(*calc_fft(e10_sim['time'], fmft_recon_x), label='FMFT Recon', alpha=0.5, c='grey')
    axs[1][i].plot(*calc_fft(e10_sim['time'], fmft_recon_y), alpha=0.5, c='grey')

    axs[0][i].plot(*calc_fft(e10_sim['time'], psi_decoupled[i]), label='Decoupled')
    axs[1][i].plot(*calc_fft(e10_sim['time'], psi_decoupled[i+action_angle_tools.N]))

    axs[0][i].plot(*calc_fft(e10_sim['time'], psi_cancelled[i]), label='Cancelled', alpha=0.8)
    axs[1][i].plot(*calc_fft(e10_sim['time'], psi_cancelled[i+action_angle_tools.N]), alpha=0.8)

    axs[0][i].set_xlim(omega_vec[i] * action_angle_tools.TO_ARCSEC_PER_YEAR - 2, omega_vec[i] * action_angle_tools.TO_ARCSEC_PER_YEAR + 2)
    axs[1][i].set_xlim(omega_vec[i+action_angle_tools.N] * action_angle_tools.TO_ARCSEC_PER_YEAR - 2, omega_vec[i+action_angle_tools.N] * action_angle_tools.TO_ARCSEC_PER_YEAR + 2)
axs[0][0].set_ylabel("Eccentricity")
axs[1][0].set_ylabel("Inclination")
axs[0][0].legend()
plt.show()
# %%
import scipy.signal

fourier = np.fft.fft(psi_decoupled[10] * np.hanning(e10_sim['time'].shape[-1]))
fourier = np.fft.fftshift(fourier)
freq = np.fft.fftfreq(e10_sim['time'].shape[-1], d=e10_sim['time'][1]) * action_angle_tools.TO_ARCSEC_PER_YEAR * 2 * np.pi
freq = np.fft.fftshift(freq)
fourier_amp = np.abs(fourier)

peaks, _ = scipy.signal.find_peaks(fourier_amp, prominence=0.005)

plt.plot(freq, fourier_amp)
plt.scatter(freq, fourier_amp, s=4)
plt.plot(freq[peaks], fourier_amp[peaks], "x")
plt.xlim(-20,-15)

peaks = peaks[np.argsort(fourier_amp[peaks])]

fmft = {'Mercury_X':{}}
for peak in peaks:
    fmft['Mercury_X'][freq[peak]*2*np.pi] = fourier[peak] / e10_sim['time'].shape[-1] / np.mean(np.hanning(e10_sim['time'].shape[-1]))
# %%
# plt.plot(np.sum([amp * np.exp(1j*freq*e10_sim['time']) for freq,amp in planet_fmft['Mercury_X'].items()],axis=0))
plt.plot(psi_decoupled[8])
# plt.plot(np.fft.ifft(fourier) / np.hanning(e10_sim['time'].shape[-1]))
plt.plot(np.sum([amp * np.exp(1j*freq*e10_sim['time']) for freq,amp in planet_fmft['Mercury_X'].items()],axis=0))
plt.plot(np.sum([amp * np.exp(1j*freq*e10_sim['time']) for freq,amp in fmft['Mercury_X'].items()],axis=0))
# %%
plt.plot(psi_decoupled[10] * np.conj(psi_decoupled[10]))
plt.plot(psi_decoupled[11] * np.conj(psi_decoupled[11]))
plt.plot(psi_decoupled[10] * np.conj(psi_decoupled[10]) + psi_decoupled[11] * np.conj(psi_decoupled[11]))

# planet_fmft = action_angle_tools.get_planet_fmft(action_angle_tools.psi_planet_list, e10_sim['time'], psi_decoupled, N=14, fmft_alg="default", display=True)
# earth_y = np.sum([amp * np.exp(1j*freq*e10_sim['time']) for freq,amp in planet_fmft["Earth_Y"].items()],axis=0)
# mars_y = np.sum([amp * np.exp(1j*freq*e10_sim['time']) for freq,amp in planet_fmft["Mars_Y"].items()],axis=0)

earth_y = psi_cancelled[10]
mars_y = psi_cancelled[11]

plt.plot(earth_y * np.conj(earth_y))
plt.plot(mars_y * np.conj(mars_y))
plt.plot(earth_y * np.conj(earth_y) + mars_y * np.conj(mars_y))
# %%
X, C_opts, found_combs = action_angle_tools.pca_combs(psi_decoupled)

gamma_0 = np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])
C_ecc = gamma_0 @ X

gamma_1 = np.array([0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0])
C_inc = gamma_1 @ X

gamma_opt = found_combs[-1]
C_opt = C_opts[-1]
# %%
# action_angle_tools.plot_action(psi[0], e10_sim['time'], take_mag=True, label=r"$\Psi$")
# action_angle_tools.plot_action(psi_decoupled[0], e10_sim['time'], take_mag=True, label=r"$\Psi_{decoupled}$")
# action_angle_tools.plot_action(psi_cancelled[0], e10_sim['time'], take_mag=True, label=r"$\Psi_{cancelled}$")
# action_angle_tools.plot_action(C_ecc, e10_sim['time'], take_mag=False, label="$C_{ecc}$")
action_angle_tools.plot_action(C_inc , e10_sim['time'], take_mag=False, label="$C_{inc}$")
# plt.xlim(0,0.6e8)
# action_angle_tools.plot_action(C_opt , e10_sim['time'], take_mag=False, label="$C_{opt}$")
plt.legend()
# %%
psi_comb = C_opts[-5:-1]
# psi_comb = [C_opts[-1]]
for i in range(len(psi_comb)):
    n = e10_sim['time'].shape[-1]
    # h = np.hanning(n) + 0.01
    h = np.ones(n)
    fourier = np.fft.fft(psi_comb[i] * h)
    fourier[1:n//2] = 0

    psi_comb[i] = np.fft.ifft(fourier) / np.mean(h) / h
psi_comb = np.array(psi_comb)
psi_comb -= np.broadcast_to(psi_comb.mean(axis=1)[None].T, psi_comb.shape)
psi_comb /= psi_comb[0,0]

# for i in range(len(psi_comb)):
#     plt.plot(e10_sim['time'], C_opts[-1])
#     p = psi_comb[i] + np.conj(psi_comb[i])
#     plt.plot(e10_sim['time'], p - p.mean()/2)
# %%
planet_fmft = action_angle_tools.get_planet_fmft(["p_"+str(i) for i in range(len(psi_comb))], e10_sim['time'], psi_comb, N=14, fmft_alg="default", display=True)
# %%
e10_sim_long, masses, rb_sim = action_angle_tools.load_sim(e05_dataset/"solarsystem_m1200.npz", np.load(e05_dataset/"solarsystem_m1200.npz", allow_pickle=True)['arr_0'][()]) 
psi_long = np.concat([e10_sim_long['x'], e10_sim_long['y']])
# %%
psi_long_decoupled = np.concat(((np.linalg.inv(ecc_rotation_matrix_opt_T) @ psi_long[:action_angle_tools.N]), (np.linalg.inv(inc_rotation_matrix_opt_T) @ psi_long[action_angle_tools.N:])))
psi_long_cancelled = action_angle_tools.apply_sequential_transforms(psi_long_decoupled, trans_fns)

X_long = np.real(psi_long_decoupled * np.conj(psi_long_decoupled))
C_ecc_long = gamma_0 @ X_long
C_inc_long = gamma_1 @ X_long
C_opt_long = gamma_opt @ X_long
# %%
# action_angle_tools.plot_action(psi_long[0], e10_sim_long['time'], take_mag=True, label=r"$\Psi$")
action_angle_tools.plot_action(psi_long_decoupled[0], e10_sim_long['time'], take_mag=True, label=r"$\Psi_{decoupled}$")
# action_angle_tools.plot_action(psi_long_cancelled[0], e10_sim_long['time'], take_mag=True, label=r"$\Psi_{cancelled}$")
action_angle_tools.plot_action(C_ecc_long, e10_sim_long['time'], take_mag=False, label="$C_{ecc}$")
action_angle_tools.plot_action(C_inc_long, e10_sim_long['time'], take_mag=False, label="$C_{inc}$")
# action_angle_tools.plot_action(C_opt_long, e10_sim_long['time'], take_mag=False, label="$C_{opt}$")
plt.xlim(0, 0.5e10)
plt.legend()
# %%
plt.plot(psi_long_decoupled[8] * np.conj(psi_long_decoupled[8]))
cancel = list(planet_fmft['Mercury_Y'].items())[1][1] * psi_long_decoupled[9] / list(planet_fmft['Venus_Y'].items())[0][1]
# cancel = psi_long_decoupled[9] / 3 * np.exp(1j * (-3))
plt.plot((psi_long_decoupled[8] - cancel) * np.conj(psi_long_decoupled[8] - cancel))
# %%
