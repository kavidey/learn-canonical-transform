import numpy as np
import math
from scipy.optimize import fsolve
from sympy import symbols, lambdify, I
from scipy.integrate import solve_ivp


def compose(f, n):
    # from https://stackoverflow.com/a/58850806/6454085
    def fn(x):
        for _ in range(n):
            x = f(x)
        return x
    return fn

def poisson_bracket(f, g, q, p):
    return I * f.diff(q) * g.diff(p) - I * f.diff(p) * g.diff(q)

def hamiltonian_system(hamiltonian_order=4, lie_order=3):
    # Define all the variables
	x = symbols("x")
	x_bar = symbols(r"\bar{x}")
	omega_0 = symbols(r"\omega_0", real = True)

	# Define Hamilton's equations
	def h_2n(n_val, x_val, x_bar_val, omega_0_val):
		return ((-1)**(n_val+1) * omega_0_val**2 / math.factorial(2*n_val)) * (1 / (2*omega_0_val))**n_val * (x_val + x_bar_val)**(2*n_val)

	# Define the Hamiltonian function: h 
	h = omega_0 * x * x_bar

	for i in range(2, hamiltonian_order):
		h += h_2n(i, x, x_bar, omega_0)
            
	h_f = lambdify([x, x_bar, omega_0], h)
	
	# Differentiate with respect to x_bar
	dhdx_bar = h.diff(x_bar) * I
	dhdx_bar_f = lambdify([x, x_bar, omega_0], dhdx_bar)
      
	pb = lambda f,g: poisson_bracket(f, g, x_bar, x)
	def lie_transformation(f, chi, order):
		fp = 0
		for i in range(order):
			fp += 1/math.factorial(i) * compose(lambda f: pb(f, chi), i)(f)
		return fp
      
	chi_4 = 1/(I*omega_0) * (x**3 * x_bar/48 - x*x_bar**3/48 - x_bar**4/384 + x**4/384)
	chi_6 = 1/(I*omega_0**2) *  (-x**6/15360 + x**5*x_bar/15360 + 7*x**4*x_bar**2/3072 - 7*x**2*x_bar**4/3072 - x*x_bar**5/15369 + x_bar**6/15360)
	chi = chi_4+chi_6
      
	xp = lie_transformation(x, -chi, lie_order)
	xp_f = lambdify([x, x_bar, omega_0], xp)

	return h_f, dhdx_bar_f, xp_f

def solve_for_x(h_f, H, omega_0_val):
    x0_guess = np.sqrt(H / omega_0_val)
    x_solution = fsolve(lambda x: h_f(x, x, omega_0_val)-H, x0_guess)
    return x_solution[0]

def sample_trajectory(dhdx_bar, x0, omega, t_span, t_eval):
	z0 = [x0.real, x0.imag]

	def diff_func(time, z):
		x_complex = z[0] + 1j * z[1]  
		dxdt = dhdx_bar(x_complex, x_complex.real - 1j * x_complex.imag, omega)
		return [dxdt.real, dxdt.imag]

	# Solve the ODEs
	sol = solve_ivp(diff_func, t_span, z0, t_eval=t_eval, rtol=1e-8, atol=1e-8)
	x_sol = sol.y[0] + 1j * sol.y[1]

	# Compute action variable J and action angle phi
	J_sol = np.abs(x_sol) ** 2
	phi_sol = np.angle(x_sol)

	return x_sol, J_sol, phi_sol