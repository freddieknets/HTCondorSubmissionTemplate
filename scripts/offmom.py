import sys
import numpy as np
from pathlib import Path
import time
start_time = time.time()

import xtrack as xt
import xpart as xp
import xcoll as xc


# Off-momentum lossmap script, specialised for FCC-ee
# ===================================================


num_turns = 1500
num_part  = 1000
sweep_hz  = 300
sigma_z  = 16.21e-3

machine = sys.argv[1]
colldb = sys.argv[2]
beam = 1
plane = sys.argv[3]
if len(sys.argv) > 4:
    engine = sys.argv[4]
else:
    engine = 'geant4'  # Default

if plane == 'DPpos':
    sweep = -sweep_hz
elif plane == 'DPneg':
    sweep = sweep_hz
else:
    raise ValueError("Incorrect plane!")
if engine not in ['everest', 'fluka', 'geant4']:
    raise ValueError("Incorrect engine!")


class NullProgressIndicator(xt.progress_indicator.DefaultProgressIndicator):
    """A simple empty class to not show any progress bar."""

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise
xt.progress_indicator.set_default_indicator(NullProgressIndicator)


# Load machine and collimator database
env = xt.load(machine)
line = env.lines['fccee_p_ring']
colldb = xc.CollimatorDatabase.from_yaml(colldb)


# Install collimators into line
aperture = xt.LimitEllipse(a=0.03, b=0.03)
if engine == 'everest':
    colldb.install_everest_collimators(line=line, verbose=True, apertures=aperture)
elif engine == 'fluka':
    colldb.install_fluka_collimators(line=line, verbose=True, apertures=aperture)
elif engine == 'geant4':
    colldb.install_geant4_collimators(line=line, verbose=True, apertures=aperture)


# Configure collimators
tw = line.twiss()
line.collimators.assign_optics(twiss=tw)


# Connect engine
if engine == 'fluka':
    capacity = 10*num_part
    xc.fluka.engine.capacity = capacity
    xc.fluka.engine.relative_capacity = 50
    xc.fluka.engine.start(line=line, cwd='.', clean=True, verbose=True)
elif engine == 'geant4':
    capacity = 10*num_part
    xc.geant4.engine.start(line=line, cwd='.', clean=True, verbose=True)
else:
    capacity = None


# Create particles
x_norm = np.random.normal(size=num_part)
px_norm = np.random.normal(size=num_part)
y_norm = np.random.normal(size=num_part)
py_norm = np.random.normal(size=num_part)
zeta, delta = xp.generate_longitudinal_coordinates(num_particles=num_part, particle_ref=line.particle_ref, line=line, sigma_z=sigma_z)
part = line.build_particles(x_norm=x_norm, px_norm=px_norm, y_norm=y_norm, py_norm=py_norm, zeta=zeta, delta=delta,
                            nemitt_x=colldb.nemitt_x, nemitt_y=colldb.nemitt_y, _capacity=capacity)


# Print some info of the RF sweep
rf_sweep = xc.RFSweep(line)
rf_sweep.prepare(sweep_per_turn=sweep/num_turns)
rf_sweep.info()


# # Move the line to an OpenMP context to be able to use all cores
# import xobjects as xo
# line.discard_tracker()
# line.build_tracker(_context=xo.ContextCpu(omp_num_threads=12))


# Switch on radiation
line.configure_radiation(model='quantum')


# Track!
line.scattering.enable()
line.track(part, num_turns=num_turns, time=True, with_progress=5)
line.scattering.disable()
print(f"Done sweeping RF in {line.time_last_track:.1f}s.")


# Switch off radiation
line.configure_radiation(model=None)


# # Move the line back to the default context to be able to use all prebuilt kernels for the aperture interpolation
# line.discard_tracker()
# line.build_tracker(_context=xo.ContextCpu())


# Make lossmap
start_interp = time.time()
ThisLM = xc.LossMap(line, line_is_reversed=False, part=part)
ThisLM.to_json(file=f'lossmap_B{beam}{plane}.json')
print(f"Done interpolating in {time.time()-start_interp:.1f}s")

# Save a summary of the collimator losses to a text file
ThisLM.save_summary(file=f'coll_summary_B{beam}{plane}.out')
print(ThisLM.summary)


if engine == 'fluka':
    xc.fluka.engine.stop(clean=True)
elif engine == 'geant4':
    xc.geant4.engine.stop(clean=True)
print(f"Total calculation time {time.time()-start_time}s")


# turns, counts = np.unique(part.at_turn, return_counts=True)
# print(turns, counts)
# for turn in turns:
#     mask = part.at_turn == turn
#     print(f"Turn {turn}:")
#     ss, cnts = np.unique(part.s[mask], return_counts=True)
#     print(f"  {ss}, {cnts}")
