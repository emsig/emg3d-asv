"""
Benchmarks for emg3d.solver.

See ./data/salt_create.py for the mesh- and data-creation of the salt example.

"""
import numpy as np
from emg3d import utils, solver
from os.path import join, dirname

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
    inp = {'grid': grid, 'res_x': dat['res'], 'freq': dat['freq']}
    if anisotropy == 'vti':
        model = utils.Model(res_z=3*dat['res'], **inp)
    elif anisotropy == 'tri':
        model = utils.Model(res_y=2*dat['res'], res_z=3*dat['res'], **inp)
    else:  # Default is 'iso'
        model = utils.Model(**inp)

    # Create source field.
    sfield = utils.get_source_field(grid, dat['src'], dat['freq'], 0)

    return grid, model, sfield


class SolverMemory:
    """Memory check for emg3d.solver.solver.

    Loops:
      - MG with or without BiCGSTAB.
      - Model with isotropic, VTI, or tri-axial resistivities.

    """
    param_names = ['sslsolver', 'anisotropy', ]
    params = [[True, False], ['iso', 'vti', 'tri'], ]

    def setup(self, sslsolver, anisotropy):
        self.grid, self.model, self.sfield = get_model('small', anisotropy)

    def teardown(self, sslsolver, anisotropy):
        del self.grid, self.sfield, self.model

    def peakmem_solver(self, sslsolver, anisotropy):
        solver.solver(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                cycle='F',
                sslsolver=sslsolver,
                semicoarsening=True,
                linerelaxation=True,
                verb=VERB)


class SolverTimeSSL:
    """Timing for emg3d.solver.solver.

    Loop:
    - MG with or without BiCGSTAB.

    """
    param_names = ['sslsolver', ]
    params = [[True, False], ]

    def setup(self, sslsolver):
        self.grid, self.model, self.sfield = get_model('small')

    def teardown(self, sslsolver):
        del self.grid, self.sfield, self.model

    def time_solver(self, sslsolver):
        solver.solver(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
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

    def setup(self, semicoarsening, linerelaxation):
        self.grid, self.model, self.sfield = get_model('small')

    def teardown(self, semicoarsening, linerelaxation):
        del self.grid, self.sfield, self.model

    def time_solver(self, semicoarsening, linerelaxation):
        solver.solver(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
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

    def setup(self, cycle):
        self.grid, self.model, self.sfield = get_model('small')

    def teardown(self, cycle):
        del self.grid, self.sfield, self.model

    def time_solver(self, cycle):
        solver.solver(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                cycle=cycle,
                sslsolver=False,
                semicoarsening=True,
                linerelaxation=True,
                verb=VERB)


class SmoothingTime:
    """Timing for emg3d.solver.smoothing.

    Loop:
    - ldir = 0, 1, 2, or 3.

    """
    param_names = ['ldir', 'size']
    params = [[0, 1, 2, 3], ['small', 'big']]


    def setup(self, ldir, size):
        self.grid, self.model, self.sfield = get_model(size)

    def teardown(self, ldir, size):
        del self.grid, self.sfield, self.model

    def time_smoothing(self, ldir, size):
        solver.smoothing(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                efield=self.sfield*0,
                nu=2,
                ldir=ldir)


class SmoothingMemory:
    """Memory for emg3d.solver.smoothing.

    Loop:
    - ldir = 0, 1, 2, or 3.

    """
    param_names = ['ldir', 'size']
    params = [[0, 1, 2, 3], ['small', 'big']]

    def setup(self, ldir, size):
        self.grid, self.model, self.sfield = get_model(size)

    def teardown(self, ldir, size):
        del self.grid, self.sfield, self.model

    def peakmem_smoothing(self, ldir, size):
        solver.smoothing(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                efield=self.sfield*0,
                nu=2,
                ldir=ldir)


class ResidualTime:
    """Timing for emg3d.solver.residual."""
    param_names = ['size', ]
    params = [['small', 'big'], ]

    def setup(self, size):
        self.grid, self.model, self.sfield = get_model(size)

    def teardown(self, size):
        del self.grid, self.sfield, self.model

    def time_smoothing(self, size):
        solver.residual(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                efield=self.sfield*0)


class ResidualMemory:
    """Memory for emg3d.solver.residual."""
    param_names = ['size', ]
    params = [['small', 'big'], ]

    def setup(self, size):
        self.grid, self.model, self.sfield = get_model(size)

    def teardown(self, size):
        del self.grid, self.sfield, self.model

    def peakmem_smoothing(self, size):
        solver.residual(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                efield=self.sfield*0)
