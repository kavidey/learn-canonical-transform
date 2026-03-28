# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
def interfere_freqs(t,a1,f1,p1,a2,f2,p2):
    a = 2*(a1*a2)
    f = f1-f2
    p = p1-p2
    o = a1**2 + a2**2
    return a*np.cos(t*f/(2*np.pi) + p)+o/2
# %%
t = np.linspace(0, 200, 500)
# %%
a_earth = np.array([0.00003314, 0.00001275/2, 0.00000330])
p_earth = np.array([-2.57, -2.34, -1.68])
f_earth = np.array([-18.850, -18.289, -17.178])

a_mars = np.array([0.00004060, 0.00000560, 0.00000321])
p_mars = np.array([1.18, -1.77, -2.80])
f_mars = np.array([-17.736, -17.175, -19.409])

a_mars /= a_earth[0]
a_earth /= a_earth[0]

amp_earth = a_earth * np.exp(1j*p_earth)
amp_mars = a_mars * np.exp(1j*p_mars)
# %%
sigs_earth = np.array([amp * np.exp(1j * t * freq/(2*np.pi)) for amp, freq in zip(amp_earth, f_earth)])
sigs_mars = np.array([amp * np.exp(1j * t * freq/(2*np.pi)) for amp, freq in zip(amp_mars, f_mars)])

sig_earth = np.sum(sigs_earth, axis=0)
sig_mars = np.sum(sigs_mars, axis=0)
# %%
sig_approx_earth = interfere_freqs(t, a_earth[0], f_earth[0], p_earth[0], a_earth[1], f_earth[1], p_earth[1]) + interfere_freqs(t, a_earth[0], f_earth[0], p_earth[0], a_earth[2], f_earth[2], p_earth[2])
sig_approx_mars = interfere_freqs(t, a_mars[0], f_mars[0], p_mars[0], a_mars[1], f_mars[1], p_mars[1]) + interfere_freqs(t, a_mars[0], f_mars[0], p_mars[0], a_mars[2], f_mars[2], p_mars[2])
# %%
plt.plot(t, np.abs(sig_earth)**2, label="earth")
plt.plot(t, np.abs(sig_mars)**2, label="mars")

plt.plot(t, sig_approx_earth,
         color='grey',
         linestyle='--')

plt.plot(t, sig_approx_mars,
         color='grey',
         linestyle='--')

plt.plot(t, np.abs(sig_earth) + np.abs(sig_mars), label="sum")

plt.plot(t, sig_approx_earth + sig_approx_mars,
         color='grey',
         linestyle='--')

plt.legend()
# %%
