import sys
import numpy as np
from pathlib import Path
import time
start_time = time.time()

import xtrack as xt
import xcoll as xc


# Pencil lossmap script, specialised for FCC-ee
# =============================================


num_turns = 200
# num_part = automatic: 50000 for Everest and 5000 for FLUKA and Geant4
sigma_z  = 16.21e-3

machine = sys.argv[1]
colldb = sys.argv[2]
beam = 1
plane = sys.argv[3].upper()
if len(sys.argv) > 4:
    engine = sys.argv[4]
else:
    engine = 'geant4'  # Default


if plane not in ['H', 'V']:
    raise ValueError("Incorrect plane!")
if engine not in ['everest', 'fluka', 'geant4', 'black']:
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
elif engine == 'black':
    colldb.install_black_absorbers(line=line, verbose=True, apertures=aperture)


# Configure collimators
tw = line.twiss()
line.collimators.assign_optics(twiss=tw)


# Connect engine
if engine == 'fluka':
    num_part  = 5000
    capacity = 10*num_part
    xc.fluka.engine.capacity = capacity
    xc.fluka.engine.relative_capacity = 50
    xc.fluka.engine.start(line=line, cwd='.', clean=True, verbose=True)
elif engine == 'geant4':
    num_part  = 5000
    capacity = 10*num_part
    xc.geant4.engine.start(line=line, cwd='.', clean=True, verbose=True)
else:
    num_part  = 50000
    capacity = None


# Generate initial pencil distribution on horizontal collimator
tcp  = f"tcp.{plane.lower()}.b{beam}"
part = line[tcp].generate_pencil(num_part, sigma_z=sigma_z, _capacity=capacity)


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
print(f"Done tracking in {line.time_last_track:.1f}s.")


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

