###
# This script is modified from Brown, G., & Rein, H. (2020). A Repository of Vanilla Long-term Integrations of the Solar System. Research Notes of the AAS, 4.
###

# This script sets up and integrates vanilla integrations of the Solar System.
# The simulation will run for approximately 30 hours.

# Both REBOUND with AVX512 is required
# See the rebound documentation for how to build with AVX512 instructions enabled
import rebound

import numpy as np
twopi = 2.*np.pi
from sys import argv

planetnames = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

try:
    # Original simulations have ids from -1000 to 1000
    n = int(argv[1]) # Simulation index (1001 total)
    n = int(np.arange(-1000, 1002, 2)[n]) # Simulation id
except:
    print("Need the simulation id as a command line argument.")
    exit()

filename = 'sim/solarsystem_'+("m" if n<0 else "p")+str(abs(n))+".bin"

# Attempt to restart the simulation
try:
    sim = rebound.Simulation(filename)
    sim.save_to_file(filename, step=int(5e3*twopi/sim.dt))
    print("Continuing from {0:8.3f} Myrs".format(sim.t/twopi/1e6))
except:
    # Attempt to read in initial conditions from file
    try:
        sim = rebound.Simulation('ss.bin')
    except:
        # Fall back: get initial conditions from NASA Horizons
        sim = rebound.Simulation()
        rebound.horizons.SSL_CONTEXT = 'unverified'
        sim.add(planetnames, date='2000-01-01 12:00')
        sim.save_to_file('ss.bin') # Store to file for reuse.

    # Perturb initial simulation by moving Mercury's x coordinate a tiny bit.
    au = 149597870700000 # Number of milimeters in 1 AU.
    dx = (0.38 * n)/au
    sim.particles[1].x += dx

    # We move to the centre of mass frame.
    sim.move_to_com()
    # We set the timestep of almost exactly 6.5 days. The sqrt(42) ensures we have a transcendental number.
    sim.dt = np.sqrt(42)*twopi/365.25
    # We choose the WHFast512 implementation of the Wisdom-Holman integrator
    # with a modified kernel an 17th order symplectic correctors, sped up with AVX512 instructions.
    sim.integrator = 'whfast512'
    # The following settings are important. If you are new to REBOUND, read through the
    # Advanced WHFast tutorial to understand what's going on. 
    sim.ri_whfast512.safe_mode = 0                 # combines symplectic correctors and drift steps at beginning and end of timesteps.
    sim.ri_whfast512.keep_unsynchronized = True    # allows for bit-by-bit reproducibility
    sim.ri_whfast512.gr_potential = True           # enable GR potential
    # Setup the output frequency for the SimulationArchive.
    # Note that we set up the output interval as a fixed number of timesteps (an integer) and
    # not a fixed time (a floating point number) to avoid rounding issues and unequal sampling
    sim.save_to_file(filename, step=int(5e3*twopi/sim.dt))

# Finally, we run the integration. 
# The simulation will exit if an escape or encounter occurs 
try:
    sim.integrate(5e9*twopi, exact_finish_time=False)
except rebound.Escape as esc:
    print(esc)
except rebound.Encounter as enc:
    print(enc)