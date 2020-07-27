"""
Benchmarks for emg3d.solver.

See ./data/salt_create.py for the mesh- and data-creation of the salt example.

"""

from os.path import join, dirname

import numpy as np

import emg3d

# Collect version info.
INFO = emg3d.__version__.split('.')
# Get dev-version.
if len(INFO) > 3:
    # Since b4de17e: setuptools: ?.?.?.dev?+ghash.
    INFO = [*INFO[:3], INFO[3].split('+')[0][3:]]
else:
    # Before was ?.?.?dev? (but dev? mostly not updated).
    INFO = [*INFO[:2], *INFO[2].split('dev')]
if len(INFO) == 3:
    # If release, add dev 999, so it does not fall below devs.
    INFO = [*INFO, '999']
INFO = tuple(map(int, INFO))

# Version-dependent imports.
# 59b4a89: 0.8.2.dev10  : Model frequency independent
# d8e98c0: 0.9.1.dev4   : VolumeModel.
# c9e595e: 0.9.3.dev1   : solver.solver => solver.solve
# 8c31324: 0.10.2.dev1  : utils refactor to fields, meshes, models
# 1b16955: 0.11.1.dev14 : res => property & mapping
if INFO < (0, 9, 3, 1):
    from emg3d.solver import solver as solve
else:
    from emg3d.solver import solve

if INFO < (0, 10, 2, 1):
    from emg3d.utils import Field, Model, TensorMesh, get_source_field
    if INFO >= (0, 9, 1, 4):
        from emg3d.utils import VolumeModel
else:
    from emg3d.fields import Field, get_source_field
    from emg3d.models import Model, VolumeModel
    from emg3d.meshes import TensorMesh

# Load salt example data set
DATA = np.load(join(dirname(__file__), 'data/salt_data.npz'),
               allow_pickle=True)
BIG = DATA['big'][()]
SMALL = DATA['small'][()]

# Change for debugging
VERB = 1


def get_model(size, anisotropy='iso'):
    """Create grid, model, and sfield from SMALL or BIG."""

    if size == 'big':
        dat = BIG
    elif size == 'small':
        dat = SMALL
    else:
        print(f"Error: `size` must be one of 'big', 'small'; provided: {size}")
        raise ValueError

    # Create grid.
    grid = TensorMesh([dat['hx'], dat['hy'], dat['hz']], dat['x0'])

    # Create model.
    inp = {'grid': grid}
    if INFO < (0, 11, 1, 14):  # resistivity, no mapping.
        inp['res_x'] = dat['res']
        if anisotropy in ['vti', 'tri']:
            inp['res_z'] = 3*dat['res']
            if anisotropy == 'tri':
                inp['res_y'] = 2*dat['res']

        if INFO < (0, 8, 2, 10):  # Frequency-dependent Model class.
            inp['freq'] = dat['freq']

    else:  # property+mapping instead of res.
        inp['property_x'] = dat['res']
        inp['mapping'] = 'Resistivity'
        if anisotropy in ['vti', 'tri']:
            inp['property_z'] = 3*dat['res']
            if anisotropy == 'tri':
                inp['property_y'] = 2*dat['res']

    model = Model(**inp)

    # Create source field.
    sfield = get_source_field(grid, dat['src'], dat['freq'], 0)

    return grid, model, sfield


class SolverTimeSSL:
    """Timing for emg3d.solver.solver.

    Loop:
    - MG with or without BiCGSTAB.

    """
    param_names = ['sslsolver', ]
    params = [[True, False], ]

    def setup_cache(self):
        data = {}
        grid, model, sfield = get_model('small')
        data['grid'] = grid
        data['model'] = model
        data['sfield'] = sfield
        return data

    def time_solver(self, data, sslsolver):
        grid = data['grid']
        model = data['model']
        sfield = Field(grid, data['sfield'])
        solve(grid=grid,
              model=model,
              sfield=sfield,
              cycle='F',
              sslsolver=sslsolver,
              semicoarsening=True,
              linerelaxation=True,
              verb=VERB)


class SolverTimeMG:
    """Timing for emg3d.solver.solver.

    Loop:
    - MG with or without semicoarsening.
    - MG with or without line relaxation.

    """
    param_names = ['semicoarsening', 'linerelaxation']
    params = [[True, False], [True, False]]

    def setup_cache(self):
        data = {}
        grid, model, sfield = get_model('small')
        data['grid'] = grid
        data['model'] = model
        data['sfield'] = sfield
        return data

    def time_solver(self, data, semicoarsening, linerelaxation):
        grid = data['grid']
        model = data['model']
        sfield = Field(grid, data['sfield'])
        solve(grid=grid,
              model=model,
              sfield=sfield,
              cycle='F',
              sslsolver=False,
              semicoarsening=semicoarsening,
              linerelaxation=linerelaxation,
              verb=VERB)


class SolverTimeCycle:
    """Timing for emg3d.solver.solver.

    Loop:
    - MG with 'V', 'W', or 'F' cycle.

    """
    param_names = ['cycle', ]
    params = [['V', 'W', 'F'], ]

    def setup_cache(self):
        data = {}
        grid, model, sfield = get_model('small')
        data['grid'] = grid
        data['model'] = model
        data['sfield'] = sfield
        return data

    def time_solver(self, data, cycle):
        grid = data['grid']
        model = data['model']
        sfield = Field(grid, data['sfield'])
        solve(grid=grid,
              model=model,
              sfield=sfield,
              cycle=cycle,
              sslsolver=False,
              semicoarsening=True,
              linerelaxation=True,
              verb=VERB)


class SmoothingTime:
    """Timing for emg3d.solver.smoothing.

    Loop:
    - lr_dir = 0, 1, 2, or 3.

    """
    param_names = ['lr_dir', 'size']
    params = [[0, 1, 2, 3], ['small', 'big']]

    def setup_cache(self):
        data = {}
        for size in self.params[1]:
            data[size] = {}
            grid, model, sfield = get_model(size)
            data[size]['grid'] = grid
            data[size]['sfield'] = sfield
            # Needs VolumeModel from 0.9.1dev4 / d8e98c0 onwards.
            if INFO < (0, 9, 1, 4):
                data[size]['model'] = model
            else:
                data[size]['model'] = VolumeModel(grid, model, sfield)
        return data

    def time_smoothing(self, data, lr_dir, size):
        grid = data[size]['grid']
        model = data[size]['model']
        sfield = Field(grid, data[size]['sfield'])
        efield = Field(grid)
        inp = (grid, model, sfield, efield, 2, lr_dir)
        emg3d.solver.smoothing(*inp)
        _ = emg3d.solver.residual(grid, model, sfield, efield)


class ResidualTime:
    """Timing for emg3d.solver.residual."""
    param_names = ['size', ]
    params = [['small', 'big'], ]

    def setup_cache(self):
        data = {}
        for size in self.params[0]:
            data[size] = {}
            grid, model, sfield = get_model(size)
            data[size]['grid'] = grid
            data[size]['sfield'] = sfield
            # Needs VolumeModel from 0.9.1dev4 / d8e98c0 onwards.
            if INFO < (0, 9, 1, 4):
                data[size]['model'] = model
            else:
                data[size]['model'] = VolumeModel(grid, model, sfield)
        return data

    def time_residual(self, data, size):
        grid = data[size]['grid']
        model = data[size]['model']
        sfield = Field(grid, data[size]['sfield'])
        _ = emg3d.solver.residual(grid, model, sfield, sfield.field*0)
