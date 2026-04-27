import numpy as np
import jax
import jax.numpy as jnp
import scipy.signal
import scipy.optimize
from sklearn.decomposition import PCA
import sympy

import rebound as rb
from reboundx import constants as rbx_constants

from utils import get_simarchive_integration_results
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
from celmech.secular import LaplaceLagrangeSystem

import matplotlib.pyplot as plt

##########################################
### Consants
##########################################

TO_ARCSEC_PER_YEAR = 60*60*180/np.pi * (2*np.pi)
TO_YEAR = 1/(2*np.pi)
N = 8
planets = ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")
psi_planet_list = (tuple(map(lambda pl: pl+"_X", planets)) + tuple(map(lambda pl: pl+"_Y", planets)))
true_omega_vec = np.array([5.535, 7.437, 17.357, 17.905, 4.257, 28.77, 3.088, 0.671] + [-5.624, -7.082, -18.837, -17.749, 0.0, -26.348, -2.993, -0.692])

# Resonances from Timescales of Chaos in the Inner Solar System: Lyapunov Spectrum and Quasi-integrals of Motion
# Order: [g1,g2,g3,g4,g5,g6,g7,g8, s1,s2,s3,s4,s5,s6,s7,s8]
harmonics = np.array([
    [0,0, 1,-1,0,0,0,0,  0,0,-1, 1,0,0,0,0],   # 1: g3 - g4 - s3 + s4
    [1,-1,0,0,0,0,0,0,  1,-1,0,0,0,0,0,0],     # 2: g1 - g2 + s1 - s2
    [0,1,0,0,-1,0,0,0, -2, 2,0,0,0,0,0,0],     # 3: g2 - g5 -2s1 +2s2
    [0,0, 2,-2,0,0,0,0,  0,0,-1, 1,0,0,0,0],   # 4: 2g3 -2g4 - s3 + s4
    [1,0,0,0,-1,0,0,0, -1, 1,0,0,0,0,0,0],     # 5
    [0,1,0,-1,0,0,0,0,  0,1,0,-1,0,0,0,0],     # 6
    [1,-2,0,1,0,0,0,0,  1,-2,0,1,0,0,0,0],     # 7
    [1,0,-1,0,0,0,0,0,  0,1,-1,0,0,0,0,0],     # 8
    [1,0,1,-2,0,0,0,0,  0,1,-1,0,0,0,0,0],     # 9
    [0,0, 3,-3,0,0,0,0,  0,0,-1, 1,0,0,0,0],   # 10
    [0,1,-1,0,0,0,0,0, -1, 2,-1,0,0,0,0,0],    # 11
    [1,0,-2,1,0,0,0,0,  0,1,0,-1,0,0,0,0],     # 12
    [2,0,-1,0,-1,0,0,0, 0,1,0,-1,0,0,0,0],     # 13
    [0,0,0,1,-1,0,0,0,  0,-1,2,-1,0,0,0,0],    # 14
    [1,0,-2,1,0,0,0,0,  1,0,1,-2,0,0,0,0],     # 15
    [1,0,0,-1,0,0,0,0,  1,0,0,-1,0,0,0,0],     # 16
    [1,-2,0,0,1,0,0,0,  3,-3,0,0,0,0,0,0],     # 17
    [1,0,0,-1,0,0,0,0,  0,1,-1,0,0,0,0,0],     # 18
    [3,-1,0,-1,-1,0,0,0, 1,0,-1,0,0,0,0,0],    # 19
    [2,-1,-1,0,0,0,0,0, 1,0,-1,0,0,0,0,0],     # 20
    [2,-1,0,-1,0,0,0,0, 1,0,-1,0,0,0,0,0],     # 21
    [0,0, 3,-3,0,0,0,0,  0,0, 2,-2,0,0,0,0],   # 22
    [2,-1,-1,1,0,0,0,0, 1,0,0,-1,0,0,0,0],     # 23
    [0,0, 2,-1,-1,0,0,0, -1,0,0,1,0,0,0,0],    # 24
    [1,0,-3,2,0,0,0,0,  0,1,0,-1,0,0,0,0],     # 25
    [1,-1,-1,1,0,0,0,0, 1,-1,0,0,0,0,0,0],     # 26
    [1,0,1,-2,0,0,0,0,  1,0,0,-1,0,0,0,0],     # 27
    [1,1,0,0,-2,0,0,0, -3, 3,0,0,0,0,0,0],     # 28
    [3,-1,0,-1,-1,0,0,0, 0,1,-1,0,0,0,0,0],    # 29
    [2,0,0,-1,-1,0,0,0, 0,1,0,-1,0,0,0,0],     # 30
])

##########################################
### General Utils
##########################################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def closest_key_entry(d, target):
    """
    Given a dictionary `d` with float keys and a target float `target`,
    returns a tuple (key, value) where the key is the one in `d`
    closest to `target`.

    Parameters
    ----------
    d : dict
        Dictionary with float keys.
    target : float
        The float to compare keys against.

    Returns
    -------
    tuple
        The (key, value) pair whose key is closest to `target`.
    """
    closest_key = min(d.keys(), key=lambda k: abs(k - target))
    return closest_key, d[closest_key]

def load_sim(path, results=None, filter_freq=None):
    dataset_path = path.parents[0]
    rb_sim = rb.Simulation(str((dataset_path/'../ss.bin').resolve()))
    masses = np.array(list(map(lambda ps: ps.m, rb_sim.particles))[1:], dtype=np.float64)
    if not results:
        results = get_simarchive_integration_results(str(path), coordinates='heliocentric')
    
    m = masses[..., None].repeat(results['a'].shape[1], axis=-1)
    G = 1
    beta = ((1 * m) / (1 + m))
    mu = G * (1 + m)
    results['Lambda'] = beta * np.sqrt(mu * np.abs(results['a']))
    
    M = results['l'] - results['pomega']
    results['lambda'] = M + results['pomega']

    results['x'] = np.sqrt(results['Lambda']) * np.sqrt(1 - np.sqrt(1-np.clip(results['e'], 0, 1)**2)) * np.exp(1j * results['pomega'])
    results['y'] = np.sqrt(2 * results['Lambda']) * np.power(1-np.clip(results['e'], 0, 1)**2, 1/4) * np.sin(results['inc']/2) * np.exp(1j * results['Omega'])

    # coordinate pairs are:
    # - Lambda, Lambda
    # - x, -i * x_bar
    # - y, -i * y_bar

    fs_arcsec_per_yr = (TO_ARCSEC_PER_YEAR / np.gradient(results['time']).mean()) * 2 * np.pi

    if filter_freq:
        b, a = scipy.signal.butter(10, filter_freq, 'low', fs=fs_arcsec_per_yr) # type: ignore[reportUnknownVariableType]
        results['x'] = scipy.signal.lfilter(b, a, results['x'])
        results['y'] = scipy.signal.lfilter(b, a, results['y'])

        for key in results.keys():
            results[key] = results[key][..., 100:]

    return results, masses, rb_sim

def custom_fmft(time, X, Nrecon, prominence=0.0005):
    N = time.shape[-1]
    result = {}

    fourier = np.fft.fft(X * np.hanning(N))
    fourier = np.fft.fftshift(fourier)
    freq = np.fft.fftfreq(N, d=time[1]-time[0])
    freq = np.fft.fftshift(freq)
    fourier_amp = np.abs(fourier)

    peaks, _ = scipy.signal.find_peaks(fourier_amp, prominence=fourier_amp/100)
    peaks = peaks[np.argsort(fourier_amp[peaks])][::-1]
    peaks = peaks[:Nrecon]

    for peak in peaks:
        result[freq[peak]*2*np.pi] = fourier[peak] / N / np.mean(np.hanning(N))
    
    return result

def get_planet_fmft(pl_list, time, X, N=14, display=False, compareto=None, fmft_alg="default", scalar=TO_ARCSEC_PER_YEAR):
    if fmft_alg == "default":
        fmft_alg = fmft
    else:
        fmft_alg = custom_fmft
    
    planet_fmft = {}
    for i,pl in enumerate(pl_list):
        fmft_res = fmft_alg(time, X[i], N)
        planet_fmft[pl] = fmft_res
        planet_freqs = np.array(list(planet_fmft[pl].keys()))
        
        if display:
            print("")
            print(pl)
            print("-------")
            for i,f in enumerate(planet_freqs):
                print(f"{f * scalar:+07.3f} \t {np.abs(planet_fmft[pl][f]):0.8f}  ∢{np.angle(planet_fmft[pl][f]):.2f}", end='')
                if compareto:
                    ctf = list(compareto[pl].keys())[i]
                    print(f"\t\t{ctf * scalar:+07.3f} \t {np.abs(compareto[pl][ctf]):0.8f}  ∢{np.angle(compareto[pl][ctf]):.2f}", end='')
                print()
    return planet_fmft

def symmetrize_axes(axes):
    y_max = np.max(np.abs(axes.get_ylim()))
    x_max = np.max(np.abs(axes.get_xlim()))

    ax_max = np.max([x_max, y_max])

    axes.set_ylim(ymin=-ax_max, ymax=ax_max)
    axes.set_xlim(xmin=-ax_max, xmax=ax_max)

def radial_plots(psi1, psi2, sym_axes=True, pl_list=None):
    if not pl_list: pl_list = psi_planet_list
    fig, axs = plt.subplots(2,len(pl_list),figsize=(30, 5))
    for i, pl in enumerate(pl_list):
        axs[0][i].set_title(pl)
        pts = psi1[i]
        axs[0][i].plot(np.real(pts), np.imag(pts))
        axs[0][i].set_aspect('equal')
        if sym_axes: symmetrize_axes(axs[0][i])
        pts = psi2[i]
        axs[1][i].plot(np.real(pts)[100:], np.imag(pts)[100:])
        axs[1][i].set_aspect('equal')
        if sym_axes: symmetrize_axes(axs[1][i])
    
    return axs

def pprint_fmft(fmft, pl, scalar=TO_ARCSEC_PER_YEAR):
    begin = r"""\begin{tabular}{l c c}
\toprule
Frequency & Amplitude \\ \midrule"""

    end = r"""\bottomrule
\end{tabular}"""

    print(begin)

    planet_freqs = np.array(list(fmft[pl].keys()))
    for i,f in enumerate(planet_freqs):
        print(f"{f * scalar:+07.3f} & {np.abs(fmft[pl][f]):0.8f} $\\measuredangle$ {np.angle(fmft[pl][f]):.2f} \\\\")

    print(end)

##########################################
### Decoupling Modes
##########################################

def decouple_modes(x, y, rb_sim, masses, optimize=True, debug=False):
    lsys = LaplaceLagrangeSystem.from_Simulation(rb_sim)
    lsys.add_general_relativity_correction(rbx_constants.C) # add GR correction

    # calculate rotation matricies and reorder entries to attempt preserve mode order
    ecc_rotation_matrix_T, ecc_eigval = lsys.diagonalize_eccentricity()
    ecc_eigval = np.diag(ecc_eigval)
    ecc_eigguess = np.diag(lsys.Neccentricity_matrix)
    _, ecc_order = scipy.optimize.linear_sum_assignment(np.abs(ecc_eigval[..., None] - ecc_eigguess[None, ...]).T)
    ecc_rotation_matrix_T = ecc_rotation_matrix_T[:,ecc_order]

    inc_rotation_matrix_T, inc_eigval = lsys.diagonalize_inclination()
    inc_eigval = np.diag(inc_eigval)
    inc_eigguess = np.diag(lsys.Ninclination_matrix)
    _, inc_order = scipy.optimize.linear_sum_assignment(np.abs(inc_eigval[..., None] - inc_eigguess[None, ...]).T)
    inc_rotation_matrix_T = inc_rotation_matrix_T[:, inc_order]

    def objective(R, G, m):
        R = jnp.reshape(R, (N,N))

        rotation_loss = ((jnp.eye(N) - R @ R.T) ** 2).sum() + (jnp.linalg.det(R) - 1) ** 2

        Phi = R.T @ G

        J_approx = jnp.abs(Phi).mean(axis=1)
        J_loss = ((jnp.abs(Phi) - J_approx[..., None]) ** 2).sum() / jnp.pow(m, 1/2)[..., None] #/ jnp.pow(J_approx, 1/2)[..., None]
        J_loss = J_loss.sum()

        # off_diag_weight = 1 / jnp.pow(jnp.outer(J_approx, J_approx), 1/2)
        off_diag_weight = jnp.pow(jnp.outer(m, m), -1/4)
        off_diag_weight /= off_diag_weight.max()
        off_diag_weight = jnp.fill_diagonal(off_diag_weight, off_diag_weight.max(), inplace=False)
        # off_diag_loss = (((jnp.ones((N,N))-jnp.eye(N)) * R.T * off_diag_weight) ** 2).sum()
        off_diag_loss = (((R.T - jnp.eye(N)) ** 2) * off_diag_weight).sum()

        # on_diag_loss = (jnp.diag(R) - 1) ** 2
        # on_diag_loss = on_diag_loss.sum()

        loss = rotation_loss*1e-1 + J_loss*1e-2 #+ off_diag_loss * 1e-3 #+ on_diag_loss * 1e-1
        return loss

    if optimize:
        obj_and_grad_ecc = jax.jit(jax.value_and_grad(lambda R: objective(R, x, masses)))
        sol_ecc = scipy.optimize.minimize(obj_and_grad_ecc, ecc_rotation_matrix_T.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
        ecc_rotation_matrix_opt_T = sol_ecc.x.reshape(N,N)

        obj_and_grad_inc = jax.jit(jax.value_and_grad(lambda R: objective(R, y, masses)))
        sol_inc = scipy.optimize.minimize(obj_and_grad_inc, inc_rotation_matrix_T.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
        inc_rotation_matrix_opt_T = sol_inc.x.reshape(N,N)
    else:
        ecc_rotation_matrix_opt_T = ecc_rotation_matrix_T
        inc_rotation_matrix_opt_T = inc_rotation_matrix_T

    if debug:
        print("ECC:")
        print(np.linalg.det(ecc_rotation_matrix_opt_T))
        print(ecc_rotation_matrix_opt_T @ ecc_rotation_matrix_opt_T.T)

        print("original\n", ecc_rotation_matrix_T)
        print("optimized\n", ecc_rotation_matrix_opt_T)
        print()


        print("INC:")
        print(np.linalg.det(inc_rotation_matrix_opt_T))
        print(inc_rotation_matrix_opt_T @ inc_rotation_matrix_opt_T.T)

        print("original\n", inc_rotation_matrix_T)
        print("optimized\n", inc_rotation_matrix_opt_T)
        print()
    
    return (
        (np.linalg.inv(ecc_rotation_matrix_opt_T) @ x),
        (np.linalg.inv(inc_rotation_matrix_opt_T) @ y)
    ), ecc_rotation_matrix_opt_T, inc_rotation_matrix_opt_T

##########################################
### Frequency Cancelling
##########################################

def get_k_vecs(order, pl_idx, skip_idx, N, include_negative=False):
    assert order % 2 == 1, "Order must be odd"
    possible_k = []

    # FIRST ORDER
    if order == 1:
        for a in range(N*2):
            if a == pl_idx:
                continue
            k = np.zeros(N*2, dtype=int)
            k[a] = 1
            possible_k.append(k)
    
    # THIRD ORDER
    if order == 3:
        for a in range(N*2):
            for b in range(a,N*2):
                for c in range(N*2):
                    if c==a:
                        continue
                    if c==b:
                        continue
                    if a in skip_idx or b in skip_idx or c in skip_idx:
                        continue
                    k = np.zeros(N*2, dtype=int)
                    k[a] +=1
                    k[b] +=1
                    k[c] -=1
                    possible_k.append(k)
    
    # FIFTH ORDER
    if order == 5:
        for a in range(N*2):
            for b in range(a,N*2):
                for c in range(b, N*2):
                    for d in range(N*2):
                        for e in range(d,N*2):
                            if a==b or a==c or a==d or a==e:
                                continue
                            if a in skip_idx or b in skip_idx or c in skip_idx or d in skip_idx or e in skip_idx:
                                continue
                            k = np.zeros(N*2, dtype=int)
                            k[a] +=1
                            k[b] -=1
                            k[c] -=1
                            k[d] -=1
                            k[e] -=1
                            possible_k.append(k)
        for a in range(N*2):
            for b in range(a,N*2):
                for c in range(b, N*2):
                    for d in range(N*2):
                        for e in range(d,N*2):
                            if d==a or d==b or d==c:
                                continue
                            if e==a or e==b or e==c:
                                continue
                            if a in skip_idx or b in skip_idx or c in skip_idx or d in skip_idx or e in skip_idx:
                                continue
                            k = np.zeros(N*2, dtype=int)
                            k[a] +=1
                            k[b] +=1
                            k[c] +=1
                            k[d] -=1
                            k[e] -=1
                            possible_k.append(k)

    # SEVENTH ORDER
    if order == 7:
        for a in range(N*2):
            for b in range(a,N*2):
                for c in range(b, N*2):
                    for d in range(c, N*2):
                        for e in range(N*2):
                            for f in range(e,N*2):
                                for g in range(f,N*2):
                                    if f==a or f==b or f==c or f==d:
                                        continue
                                    if g==a or g==b or g==c or g==d:
                                        continue
                                    if a in skip_idx or b in skip_idx or c in skip_idx or d in skip_idx or e in skip_idx or f in skip_idx or g in skip_idx:
                                        continue
                                    k = np.zeros(N*2, dtype=int)
                                    k[a] +=1
                                    k[b] +=1
                                    k[c] +=1
                                    k[d] +=1
                                    k[e] -=1
                                    k[f] -=1
                                    k[g] -=1
                                    possible_k.append(k)

    possible_k = np.array(possible_k)
    if include_negative:
        possible_k = np.concat((possible_k, -possible_k), axis=0)
    return possible_k

def get_combs(order, pl_fmft, pl_list, omega_vec, display=False, include_negative=False, omega_pct_thresh=1e-4, omega_abs_thresh=1e-3, skip_idx=[], min_freq=0.0, N=N, fmft_scalar=TO_ARCSEC_PER_YEAR):
    combs = []
    for i,pl in enumerate(pl_list):
        if display:
            print()
            print(f"{pl} \t base amp: {np.abs(list(pl_fmft[pl].items())[0][1]):.2g}") 
            print("-"*len(pl))
            print("kvec \t\t\t\t\t omega \t err. \t amplitude")
        comb = {}
        for k in get_k_vecs(order, i, skip_idx, N, include_negative=include_negative):
            omega = k @ omega_vec

            if order != 1 and (np.abs(omega_vec/omega-1) < 1e-3).any():
                continue

            if np.abs(omega) < min_freq:
                continue

            omega_N,amp = closest_key_entry(pl_fmft[pl],omega)
            omega_pct_error = np.abs(omega_N/omega-1)
            omega_abs_error = np.abs(omega_N - omega)

            # print("testing", k, omega, omega_N, omega_abs_error)

            # if the frequency is close to a frequency that exists in the planet
            if omega_pct_error<omega_pct_thresh and omega_abs_error < omega_abs_thresh:
                # check if we already found a kvec that matches this frequency
                omega_N_exists = False
                to_del = []
                for old_k,(_, old_omega, old_err) in comb.items():
                    if old_omega == omega_N:
                        omega_N_exists = True
                        # if our new kvec is better than the old one, delete it
                        if old_err > omega_pct_error:
                            # del comb[old_k]
                            to_del.append(old_k)
                            omega_N_exists = False
                for d in to_del:
                    del comb[d]
                # add new kvec if it is better or there wasn't an existing one with the same frequency
                if not omega_N_exists:
                    comb[tuple(k)] = (amp, omega_N, omega_pct_error)
            
        if display:
            for k,(amp, omega, err) in comb.items():
                k = np.array(k)
                print(k,"\t{:+07.3f}\t{:.1g},\t{:.1g}".format(omega*fmft_scalar,err,np.abs(amp)))
        combs.append(comb)
    return combs

def eval_transform(x_bars, subs, N=N):
    x_i_lambda = [sympy.lambdify(x_bars[-2], x_bars[-1][i].subs(subs)) for i in range(N*2)]

    trans = lambda x: np.array([x_lambda(*x) for x_lambda in x_i_lambda])
    return trans

def apply_sequential_transforms(x, transforms):
    for transform in reversed(transforms):
        x = transform(x)
    return x

def cancel_frequencies(psi, time, omega_vec, omega_amp, iterations, omega_pct_thresh=2e-5, omega_abs_thresh=1e-3, n_fmft=14, fmft_alg="default", min_freq=0.0, debug=False, psi_planet_list=psi_planet_list, fmft_scalar=TO_ARCSEC_PER_YEAR, epsilon_thresh=4, initial_fmft=None):
    N = len(psi_planet_list)//2
    # Calculate all planet FMFTs
    if debug: print("FMFT Results")
    if initial_fmft:
        planet_fmft = initial_fmft
    else:
        print(n_fmft, fmft_alg, debug, fmft_scalar)
        planet_fmft = get_planet_fmft(psi_planet_list, time, psi, N=n_fmft, fmft_alg=fmft_alg, display=debug, scalar=fmft_scalar)

    skip_planet_idx = []
    if np.min(np.abs(omega_vec)) < 0.001:
        skip_planet_idx.append(int(np.argmin(np.abs(omega_vec))))

    # Figure out which planets we can use to build combinations
    if debug: print("planets that don't satisfy epsilon assumption")
    for i, pl in enumerate(psi_planet_list):
        amps = [v for k, v in planet_fmft[pl].items()]
        amp_ratio = np.abs(amps[0]) / np.abs(amps[1])
        if amp_ratio < epsilon_thresh:
            print(pl, amp_ratio)
            skip_planet_idx.append(i)
    if debug: skip_planet_idx.sort()

    # Build and cancel according to input iterations
    x_val = psi
    x = [sympy.Symbol("X_"+str(i)) for i in range(N*2)]
    x_bar_0 = [sympy.Symbol("\\bar X^{(0)}_"+str(i)) for i in range(N*2)]

    x_bars = [x_bar_0]
    subs = {x_bar_0[i]: x[i] for i in range(N*2)}

    trans_fns = []

    for i,order in enumerate(iterations):
        if debug: print("#"*10, f"ITERATION {i+1} - ORDER {order}", "#"*10)
        last_x_val = apply_sequential_transforms(x_val, trans_fns)
        if i == 0:
            last_fmft = planet_fmft
        else:
            last_fmft = get_planet_fmft(psi_planet_list, time, last_x_val, n_fmft, fmft_alg=fmft_alg, display=debug, scalar=fmft_scalar)
        combs = get_combs(order, last_fmft, psi_planet_list, omega_vec, display=debug, include_negative=False, omega_pct_thresh=omega_pct_thresh, omega_abs_thresh=omega_abs_thresh, skip_idx=skip_planet_idx, min_freq=min_freq, N=N, fmft_scalar=fmft_scalar)

        x_bar_i = [sympy.Symbol(f"\\bar X^{{({i+1})}}_"+str(j)) for j in range(N*2)]

        # loop through each object
        for j in range(N*2):
            # to first order the coordinate is the original coordinate
            x_bar_i_j = x_bars[-1][j]
            # correct for each combination
            for k,(amp, omega, _) in combs[j].items():
                term = amp
                # loop through each object
                delta = 0
                for k_idx in range(N*2):
                    # add each object the correct number of times
                    for l in range(np.abs(k[k_idx])):
                        term *= x_bars[-1][k_idx]/omega_amp[k_idx] if k[k_idx] > 0 else x_bars[-1][k_idx].conjugate()/np.conj(omega_amp[k_idx])
                x_bar_i_j -= term
            subs[x_bar_i[j]] = x_bar_i_j
        x_bars.append(x_bar_i)
        trans_fns.append(eval_transform(x_bars, subs, N))

    psi_trans = apply_sequential_transforms(x_val, trans_fns)

    return psi_trans, trans_fns, combs

##########################################
### Linear Combinations
##########################################
def pca_combs(psi):
    X = np.real(psi * np.conj(psi))
    pca = PCA().fit(X.T / X.mean())


    found_combs = []
    C_opts = []
    for i in reversed(range(1, 6)):
        gamma_n = pca.components_[-i] / (np.max(np.abs(pca.components_[-i])) / 2)
        gamma_n = gamma_n.round()
        found_combs.append(gamma_n)
        C_opt = gamma_n @ X
        C_opts.append(C_opt)
    
    return X, C_opts, found_combs

def plot_action(x, t, ax=None, take_mag=True, **kwargs):
    if take_mag: x = np.real(x * np.conj(x))
    
    action = x / x[0]
    action = action - action[:action.shape[0]//2].mean()

    if ax:
        ax.plot(t, action, **kwargs)
    else:
        plt.plot(t, action, **kwargs)