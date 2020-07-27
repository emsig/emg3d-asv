"""
Benchmarks for emg3d.solver.

See ./data/salt_create.py for the mesh- and data-creation of the salt example.

"""

from os.path import join, dirname

import numpy as np

import emg3d

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
    grid = TensorMesh([dat['hx'], dat['hy'], dat['hz']], dat['x0'])

    # Create model.
    try:  # from 0.12.0 onwards, use property with mapping instead of res.
        inp = {'grid': grid, 'property_x': dat['res'],
               'mapping': 'Resistivity'}
        if anisotropy == 'vti':
            model = Model(property_z=3*dat['res'], **inp)
        elif anisotropy == 'tri':
            model = Model(property_y=2*dat['res'], property_z=3*dat['res'],
                          **inp)
        else:  # Default is 'iso'
            model = Model(**inp)
    except VariableCatch:
        try:  # Model in emg3d < v0.9.0 was frequency dependent.
            inp = {'grid': grid, 'res_x': dat['res'], 'freq': dat['freq']}
            if anisotropy == 'vti':
                model = Model(res_z=3*dat['res'], **inp)
            elif anisotropy == 'tri':
                model = Model(res_y=2*dat['res'], res_z=3*dat['res'], **inp)
            else:  # Default is 'iso'
                model = Model(**inp)
        except VariableCatch:  # Model in emg3d > v0.9.0 is freq. independent.
            inp = {'grid': grid, 'res_x': dat['res']}
            if anisotropy == 'vti':
                model = Model(res_z=3*dat['res'], **inp)
            elif anisotropy == 'tri':
                model = Model(res_y=2*dat['res'], res_z=3*dat['res'], **inp)
            else:  # Default is 'iso'
                model = Model(**inp)

    # Create source field.
    sfield = get_source_field(grid, dat['src'], dat['freq'], 0)

    return grid, model, sfield


# Change from emg3d.solver.solver to emg3d.solver.solve.
try:
    from emg3d.solver import solve
except ImportError:
    from emg3d.solver import solver as solve

# Change from emg3d.utils to other modules.
try:
    from emg3d.utils import Field, Model, TensorMesh, get_source_field

    # Find out if we are in the before eef25f71 or not.
    grid, tmodel, sfield = get_model('small')
    try:
        try:  # Needs VolumeModel from d8e98c0 onwards.
            model = emg3d.utils.VolumeModel(grid, tmodel, sfield)
        except AttributeError:
            model = tmodel
        a, b = emg3d.solver.residual(grid, model, sfield, sfield*0)
        BEFORE = False
    except ValueError:
        BEFORE = True
    del grid, sfield, tmodel, model

except ImportError:
    BEFORE = True
    from emg3d.fields import Field, get_source_field
    from emg3d.models import Model, VolumeModel
    from emg3d.meshes import TensorMesh


class SolverMemory:
    """Memory check for emg3d.solver.solver.

    Loops:
      - MG with or without BiCGSTAB.
      - Model with isotropic, VTI, or tri-axial resistivities.

    """
    param_names = ['sslsolver', 'anisotropy', ]
    params = [[True, False], ['iso', 'vti', 'tri'], ]

    def setup_cache(self):
        data = {}
        for anisotropy in self.params[1]:
            data[anisotropy] = {}
            grid, model, sfield = get_model('small', anisotropy)
            data[anisotropy]['grid'] = grid
            data[anisotropy]['model'] = model
            data[anisotropy]['sfield'] = sfield
        return data

    def peakmem_solver(self, data, sslsolver, anisotropy):
        grid = data[anisotropy]['grid']
        model = data[anisotropy]['model']
        sfield = Field(grid, data[anisotropy]['sfield'])
        solve(grid=grid,
              model=model,
              sfield=sfield,
              cycle='F',
              sslsolver=sslsolver,
              semicoarsening=True,
              linerelaxation=True,
              verb=VERB)


class SmoothingMemory:
    """Memory for emg3d.solver.smoothing.

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
                data[size]['model'] = VolumeModel(grid, model, sfield)
            except AttributeError:
                data[size]['model'] = model
        return data

    def peakmem_smoothing(self, data, lr_dir, size):
        grid = data[size]['grid']
        model = data[size]['model']
        sfield = Field(grid, data[size]['sfield'])
        efield = Field(grid)
        inp = (grid, model, sfield, efield, 2, lr_dir)
        if BEFORE:
            emg3d.solver.smoothing(*inp)
            res = emg3d.solver.residual(grid, model, sfield, efield)
            norm = np.linalg.norm(res)
        else:  # After, residual is included in smoothing and norm in residual.
            res, norm = emg3d.solver.smoothing(*inp)


class ResidualMemory:
    """Memory for emg3d.solver.residual."""
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
                data[size]['model'] = VolumeModel(grid, model, sfield)
            except AttributeError:
                data[size]['model'] = model
        return data

    def peakmem_residual(self, data, size):
        grid = data[size]['grid']
        model = data[size]['model']
        sfield = Field(grid, data[size]['sfield'])
        if BEFORE:
            res = emg3d.solver.residual(grid, model, sfield, sfield.field*0)
            norm = np.linalg.norm(res)
        else:  # After, norm is included in residual.
            res, norm = emg3d.solver.residual(
                    grid, model, sfield, sfield.field*0)
