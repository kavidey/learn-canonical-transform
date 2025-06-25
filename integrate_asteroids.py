# %%
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import rebound as rb
import celmech as cm

outer_only = True
print(f"Outer Planets Only: {outer_only}")
# %%
integration_path = Path("datasets") / "asteroid_integration" / ("outer_planets" if outer_only else "full_system")
integration_path.mkdir(parents=True, exist_ok=True)
# %%
# Read the table with the defined column specifications
df = pd.read_fwf('MPCORB.DAT', colspecs=[[0,7], [8,14], [15,19], [20,25], [26,35], [36,46], [47, 57], [58,68], [69,81], [82, 91], [92, 103]])
df = df[df['Epoch'] == 'K239D'] # take only ones at common epoch--almost all of them

df.infer_objects()
for c in ['a', 'e', 'Incl.', 'Node', 'Peri.', 'M']:
	df[c] = pd.to_numeric(df[c])

labels = pd.read_fwf('proper_catalog24.dat', colspecs=[[0,10], [10,18], [19,28], [29,37], [38, 46], [47,55], [56,66], [67,78], [79,85], [86, 89], [90, 97]], header=None, index_col=False, names=['propa', 'da', 'prope', 'de', 'propsini', 'dsini', 'g', 's', 'H', 'NumOpps', "Des'n"])

merged_df = pd.merge(df, labels, on="Des'n", how="inner")
# %%
sim = rb.Simulation()
date = "2023-09-13 12:00"
sim.add("Sun", date=date)
if not outer_only:
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
def run_sim(r):
    idx, row = r
    # sim = rb.Simulation('planets.bin')
    # if outer_only:
    #     for i in range(4):
    #         sim.remove(1)
    sim = rb.Simulation(str(integration_path/'planets.bin'))
    sim.add(a=row['a'], e=row['e'], inc=row['Incl.']*np.pi/180, Omega=row['Node']*np.pi/180, omega=row['Peri.']*np.pi/180, M=row['M'], primary=sim.particles[0])
    sim.move_to_com()

    ps = sim.particles
    ps[-1].a

    sim.integrator='whfast'
    sim.dt = ps[1].P/100.
    sim.ri_whfast.safe_mode = 0

    Tfin_approx = 2e7*ps[-1].P
    total_steps = np.ceil(Tfin_approx / sim.dt)
    Tfin = total_steps * sim.dt + sim.dt
    Nout = 30_000

    sim_file = integration_path / f"asteroid_integration_{row["Des'n"]}.sa"
    sim.save_to_file(str(sim_file), step=int(np.floor(total_steps/Nout)), delete_file=True)
    sim.integrate(Tfin, exact_finish_time=0)
# %%
with Pool(40) as p:
      p.map(run_sim, merged_df[:1000].iterrows())
# %%
