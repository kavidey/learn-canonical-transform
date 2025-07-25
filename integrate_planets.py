# %%
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import rebound as rb
import celmech as cm

import sys
sys.path.insert(0, 'SBDynT/src')
import sbdynt as sbd

import reboundx
from reboundx import constants
# %%
integration_path = Path("datasets") / "planet_integration"
integration_path.mkdir(parents=True, exist_ok=True)
# %%
# sim = rb.Simulation()
# date = "2023-09-13 12:00"
# sim.add("Sun", date=date)
# sim.add("Mercury", date=date)
# sim.add("Venus", date=date)
# sim.add("Earth", date=date)
# sim.add("Mars", date=date)
# sim.add("Jupiter", date=date)
# sim.add("Saturn", date=date)
# sim.add("Uranus", date=date)
# sim.add("Neptune", date=date)
# sim.save_to_file(str(integration_path/"planets.bin"))

epoch=2457019.5
sim = rb.Simulation()

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='sun',epoch=epoch)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='mercury',epoch=epoch)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='venus',epoch=epoch)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='earth',epoch=epoch)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='mars',epoch=epoch)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='jupiter',epoch=epoch)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='saturn',epoch=epoch)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='uranus',epoch=epoch)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='neptune',epoch=epoch)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

assert len(sim.particles) == 9, "Error adding planets"

sim.save_to_file(str(integration_path/"planets.bin"))
# %%
sim = rb.Simulation(str(integration_path/'planets.bin'))
sim.move_to_com()

rebx = reboundx.Extras(sim)
gr = rebx.load_force("gr_potential")
rebx.add_force(gr)
gr.params["c"] = constants.C

ps = sim.particles
# %%
sim.integrator='whfast'
# sim.dt = ps[1].P/10.
sim.dt = (1.1 / 365.25) * np.pi*2 # could set to 8 days
sim.ri_whfast.safe_mode = 0

Tfin_approx = -1e7*np.pi*2
interval = 600 * np.pi*2
Nout = abs(int(Tfin_approx/interval))

# Tfin_approx = 5e7*ps[4].P
# Tfin_approx = -1e7*np.pi*2
total_steps = np.ceil(np.abs(Tfin_approx) / sim.dt)
Tfin = total_steps * sim.dt + sim.dt
# Nout = 8_192
step = int(np.floor(total_steps/Nout))
# %%
sim_file = integration_path / f"planet_integration.{int(abs(Tfin_approx) / (2*np.pi))}.{int(Nout)}.sa"
sim.save_to_file(str(sim_file), step=step, delete_file=True)
sim.integrate(Tfin, exact_finish_time=0)
# %%
