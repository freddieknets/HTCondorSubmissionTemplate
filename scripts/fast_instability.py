import sys
import json
import numpy as np
from pathlib import Path
import time
start_time = time.time()

import xobjects as xo
import xtrack as xt
import xpart as xp
import xcoll as xc


# Fast-instability lossmap script, specialised for FCC-ee
# =======================================================

num_part = 5000

machine = sys.argv[1]
colldb = sys.argv[2]
beam = 1
plane = sys.argv[3].upper()
phase = int(sys.argv[4])
if len(sys.argv) > 5:
    engine = sys.argv[5]
else:
    engine = 'geant4'  # Default

phi = 0
sigma_z  = 16.21e-3

if plane not in ['H', 'V']:
    raise ValueError("Incorrect plane!")
if phase not in [0, 30, 60, 90]:
    raise ValueError("Incorrect phase!")
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
    capacity = 10*num_part
    xc.fluka.engine.capacity = capacity
    xc.fluka.engine.relative_capacity = 50
    xc.fluka.engine.start(line=line, cwd='.', clean=True, verbose=True)
elif engine == 'geant4':
    capacity = 10*num_part
    xc.geant4.engine.start(line=line, cwd='.', clean=True, verbose=True)
else:
    capacity = None


# Install exciters
settings = {
    'H': {
        0: {'kick_amplitude': 5e-14, 'rise_time_nturns': 5, 'n_turns': 100},
        30: {'kick_amplitude': 5e-14, 'rise_time_nturns': 5, 'n_turns': 100},
        60: {'kick_amplitude': 5e-14, 'rise_time_nturns': 5, 'n_turns': 100},
        90: {'kick_amplitude': 5e-14, 'rise_time_nturns': 5, 'n_turns': 100}
    },
    'V': {
        0: {'kick_amplitude': 1e-13, 'rise_time_nturns': 10, 'n_turns': 100},
        30: {'kick_amplitude': 1e-13, 'rise_time_nturns': 10, 'n_turns': 100},
        60: {'kick_amplitude': 1e-13, 'rise_time_nturns': 10, 'n_turns': 100},
        90: {'kick_amplitude': 3e-13, 'rise_time_nturns': 10, 'n_turns': 100}
    }
}
f_rev = 1 / tw.T_rev0
f_samp = f_rev
rise_time = settings[plane][phase]['rise_time_nturns'] * tw.T_rev0
total_time = settings[plane][phase]['n_turns'] / f_rev
time_ = np.arange(0, total_time, 1 / f_samp)
if plane == 'H':
    knl = settings[plane][phase]['kick_amplitude']
    ksl = 0
elif plane == 'V':
    knl = 0
    ksl = settings[plane][phase]['kick_amplitude']
f_ex = tw.qx * f_rev if plane == 'H' else tw.qy * f_rev
tt_sigmas = tw.get_beam_covariance(nemitt_x=colldb.nemitt_x, nemitt_y=colldb.nemitt_y)
tt = line.get_table()
names = tt.rows[f'fast_instability_marker.{plane.lower()}.{phase}.*'].name
if len(names) == 0:
    raise ValueError("No exciter markers found in the line!")
exciter_aper_placements = []
for nn in names:
    exciter_name = nn.replace("marker", "kicker")
    sigma = tt_sigmas['sigma_x', nn] if plane == 'H' else tt_sigmas['sigma_y', nn]
    samples = 1 / sigma * np.cos(2 * np.pi * f_ex * time_ + phi) * np.exp(time_ / rise_time)
    env.elements[exciter_name] = xt.Exciter(
        samples=samples,
        sampling_frequency=f_samp,
        frev=f_rev,
        start_turn=0,
        knl=[knl],
        ksl=[ksl]
    )
    line.replace(nn, exciter_name)

    # Define exciter bounding apertures
    aper_name = f'{exciter_name}_aper'
    env.elements[aper_name] = aperture
    env.new(aper_name + '..0', aper_name, mode='replica')
    exciter_aper_placements.append(env.place(f'{aper_name}..0', at=f'{exciter_name}@start'))
    env.new(aper_name + '..1', aper_name, mode='replica')
    exciter_aper_placements.append(env.place(f'{aper_name}..1', at=f'{exciter_name}@end'))
line.insert(exciter_aper_placements)


# Create particles
x_norm = np.random.normal(size=num_part)
px_norm = np.random.normal(size=num_part)
y_norm = np.random.normal(size=num_part)
py_norm = np.random.normal(size=num_part)
zeta, delta = xp.generate_longitudinal_coordinates(num_particles=num_part, particle_ref=line.particle_ref, line=line, sigma_z=sigma_z)
part = line.build_particles(x_norm=x_norm, px_norm=px_norm, y_norm=y_norm, py_norm=py_norm, zeta=zeta, delta=delta,
                            nemitt_x=colldb.nemitt_x, nemitt_y=colldb.nemitt_y, _capacity=capacity)


# # Move the line to an OpenMP context to be able to use all cores
# import xobjects as xo
# line.discard_tracker()
# line.build_tracker(_context=xo.ContextCpu(omp_num_threads=12))


# Switch on radiation
line.configure_radiation(model='quantum')


# Track!
line.scattering.enable()
line.track(part, num_turns=settings[plane][phase]['n_turns'], time=True, with_progress=5)
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
ThisLM.to_json(file=f'lossmap_B{beam}{plane}_ph{phase}.json')
print(f"Done interpolating in {time.time()-start_interp:.1f}s")

# Save a summary of the collimator losses to a text file
ThisLM.save_summary(file=f'coll_summary_B{beam}{plane}_ph{phase}.out')
print(ThisLM.summary)

# Save losses distribution over time
mask = part.state > xt.particles.LAST_INVALID_STATE
with open(f'particles_dict_B{beam}{plane}_ph{phase}.json', 'w') as fid:
    json.dump({
        'state': part.state[mask], 'at_turn': part.at_turn[mask],
        'at_element': part.at_element[mask], 's': part.s[mask],
        'energy': part.energy[mask]
    }, fid, indent=4, cls=xo.JEncoder)


if engine == 'fluka':
    xc.fluka.engine.stop(clean=True)
elif engine == 'geant4':
    xc.geant4.engine.stop(clean=True)
print(f"Total calculation time {time.time()-start_time}s")
