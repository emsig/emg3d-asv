"""
Benchmarks for emg3d.solver.

See ./data/salt_create.py for the mesh- and data-creation of the salt example.

"""
import numpy as np
from emg3d import utils, solver
from os.path import join, dirname

VariableCatch = (LookupError, AttributeError, ValueError, TypeError, NameError)

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
    grid = utils.TensorMesh([dat['hx'], dat['hy'], dat['hz']], dat['x0'])

    # Create model.
    try:  # Model in emg3d < v0.9.0 was frequency dependent.
        inp = {'grid': grid, 'res_x': dat['res'], 'freq': dat['freq']}
        if anisotropy == 'vti':
            model = utils.Model(res_z=3*dat['res'], **inp)
        elif anisotropy == 'tri':
            model = utils.Model(res_y=2*dat['res'], res_z=3*dat['res'], **inp)
        else:  # Default is 'iso'
            model = utils.Model(**inp)

    except VariableCatch:  # Model in emg3d > v0.9.0 is frequency independent.
        inp = {'grid': grid, 'res_x': dat['res']}
        if anisotropy == 'vti':
            model = utils.Model(res_z=3*dat['res'], **inp)
        elif anisotropy == 'tri':
            model = utils.Model(res_y=2*dat['res'], res_z=3*dat['res'], **inp)
        else:  # Default is 'iso'
            model = utils.Model(**inp)

    # Create source field.
    sfield = utils.get_source_field(grid, dat['src'], dat['freq'], 0)

    return grid, model, sfield


# Find out if we are in the before eef25f71 or not.
# (We also use this to change from emg3d.solver.solver to emg3d.solver.solve.
grid, tmodel, sfield = get_model('small')
try:
    try:  # Needs VolumeModel from d8e98c0 onwards.
        model = utils.VolumeModel(grid, tmodel, sfield)
    except AttributeError:
        model = tmodel
    a, b = solver.residual(grid, model, sfield, sfield*0)
    BEFORE = False
    solve = solver.solver
except ValueError:
    BEFORE = True
    solve = solver.solve
del grid, sfield, tmodel, model


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
        sfield = utils.Field(grid, data['sfield'])
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
        sfield = utils.Field(grid, data['sfield'])
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
        sfield = utils.Field(grid, data['sfield'])
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
            try:  # Needs VolumeModel from d8e98c0 onwards.
                data[size]['model'] = utils.VolumeModel(grid, model, sfield)
            except AttributeError:
                data[size]['model'] = model
        return data

    def time_smoothing(self, data, lr_dir, size):
        grid = data[size]['grid']
        model = data[size]['model']
        sfield = utils.Field(grid, data[size]['sfield'])
        efield = utils.Field(grid)
        inp = (grid, model, sfield, efield, 2, lr_dir)
        if BEFORE:
            solver.smoothing(*inp)
            res = solver.residual(grid, model, sfield, efield)
            norm = np.linalg.norm(res)
        else:  # After, residual is included in smoothing and norm in residual.
            res, norm = solver.smoothing(*inp)


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
            try:  # Needs VolumeModel from d8e98c0 onwards.
                data[size]['model'] = utils.VolumeModel(grid, model, sfield)
            except AttributeError:
                data[size]['model'] = model

        return data

    def time_residual(self, data, size):
        grid = data[size]['grid']
        model = data[size]['model']
        sfield = utils.Field(grid, data[size]['sfield'])
        if BEFORE:
            res = solver.residual(grid, model, sfield, sfield.field*0)
            norm = np.linalg.norm(res)
        else:  # After, norm is included in residual.
            res, norm = solver.residual(grid, model, sfield, sfield.field*0)
