# %%
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import rebound as rb
import celmech as cm

# %%
integration_path = Path("datasets") / "planet_integration"
integration_path.mkdir(parents=True, exist_ok=True)
# %%
sim = rb.Simulation()
date = "2023-09-13 12:00"
sim.add("Sun", date=date)
sim.add("Mercury", date=date)
sim.add("Venus", date=date)
sim.add("Earth", date=date)
sim.add("Mars", date=date)
sim.add("Jupiter", date=date)
sim.add("Saturn", date=date)
sim.add("Uranus", date=date)
sim.add("Neptune", date=date)
sim.save_to_file(str(integration_path/"planets.bin"))
# %%
sim = rb.Simulation(str(integration_path/'planets.bin'))
sim.move_to_com()

ps = sim.particles

sim.integrator='whfast'
sim.dt = ps[1].P/100.
sim.ri_whfast.safe_mode = 0

Tfin_approx = 5e8*ps[-1].P
total_steps = np.ceil(Tfin_approx / sim.dt)
Tfin = total_steps * sim.dt + sim.dt
Nout = 100_000

sim_file = integration_path / f"planet_integration.sa"
sim.save_to_file(str(sim_file), step=int(np.floor(total_steps/Nout)), delete_file=True)
sim.integrate(Tfin, exact_finish_time=0)
# %%
