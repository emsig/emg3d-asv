import numpy as np
from emg3d import utils, solver
from os.path import join, dirname

# Load data.
DATA = np.load(join(dirname(__file__), 'data/salt_data.npz'),
               allow_pickle=True)
RES = DATA['res'][()]
MESH = DATA['mesh'][()]


class SolverMemory:
    """Timing for emg3d.solver.solver.

    See data/salt_create.py for the mesh- and data-creation.

    Memory:
      - MG/MG with BiCGSTAB
      - iso/vti/tri-axial

    """

    # Parameters to loop over
    params = [[True, False], ['iso', 'vti', 'tri'], ]
    param_names = ['sslsolver', 'anisotropy', ]

    def setup(self, sslsolver, anisotropy):

        # Create grid.
        self.grid = utils.TensorMesh(
                [MESH['hx'], MESH['hy'], MESH['hz']], MESH['x0'])

        # Get source field.
        self.sfield = utils.get_source_field(
                self.grid, MESH['src'], MESH['freq'], 0)

        # Create model.
        inp = {'grid': self.grid, 'res_x': RES, 'freq': MESH['freq']}
        if anisotropy == 'iso':
            self.model = utils.Model(**inp)
        elif anisotropy == 'vti':
            self.model = utils.Model(res_z=3*RES, **inp)
        else:
            self.model = utils.Model(res_y=2*RES, res_z=3*RES, **inp)

    def teardown(self, sslsolver, anisotropy):
        del self.grid, self.sfield, self.model

    def peakmem_solver(self, sslsolver, anisotropy):
        solver.solver(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                sslsolver=sslsolver,
                semicoarsening=True,
                linerelaxation=True,
                verb=2)


class SolverTimeSSL:
    """Timing for emg3d.solver.solver.

    See data/salt_create.py for the mesh- and data-creation.

    Time:
    - Timing, with/without sslsolver.

    """

    # Parameters to loop over
    params = [[True, False], ]
    param_names = ['sslsolver', ]

    def setup(self, sslsolver):

        # Create grid.
        self.grid = utils.TensorMesh(
                [MESH['hx'], MESH['hy'], MESH['hz']], MESH['x0'])

        # Get source field.
        self.sfield = utils.get_source_field(
                self.grid, MESH['src'], MESH['freq'], 0)

        # Create model.
        inp = {'grid': self.grid, 'res_x': RES, 'freq': MESH['freq']}
        self.model = utils.Model(**inp)

    def teardown(self, sslsolver):
        del self.grid, self.sfield, self.model

    def time_solver(self, sslsolver):
        solver.solver(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                sslsolver=sslsolver,
                semicoarsening=True,
                linerelaxation=True,
                verb=2)


class SolverTimeMG:
    """Timing for emg3d.solver.solver.

    See data/salt_create.py for the mesh- and data-creation.

    Time:
    - Timing, MG with/without semicoarsening; with/without line relaxation

    """

    # Parameters to loop over
    params = [[True, False], [True, False]]
    param_names = ['semicoarsening', 'linerelaxation']

    def setup(self, semicoarsening, linerelaxation):

        # Create grid.
        self.grid = utils.TensorMesh(
                [MESH['hx'], MESH['hy'], MESH['hz']], MESH['x0'])

        # Get source field.
        self.sfield = utils.get_source_field(
                self.grid, MESH['src'], MESH['freq'], 0)

        # Create model.
        inp = {'grid': self.grid, 'res_x': RES, 'freq': MESH['freq']}
        self.model = utils.Model(**inp)

    def teardown(self, semicoarsening, linerelaxation):
        del self.grid, self.sfield, self.model

    def time_solver(self, semicoarsening, linerelaxation):
        solver.solver(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                semicoarsening=semicoarsening,
                linerelaxation=linerelaxation,
                verb=2)


class SolverTimeCycle:
    """Timing for emg3d.solver.solver.

    See data/salt_create.py for the mesh- and data-creation.

    Time:
    - Timing, MG all cycles; semicoarsening and line relaxation

    """

    # Parameters to loop over
    params = [['V', 'W', 'F'], ]
    param_names = ['cycle', ]

    def setup(self, cycle):

        # Create grid.
        self.grid = utils.TensorMesh(
                [MESH['hx'], MESH['hy'], MESH['hz']], MESH['x0'])

        # Get source field.
        self.sfield = utils.get_source_field(
                self.grid, MESH['src'], MESH['freq'], 0)

        # Create model.
        inp = {'grid': self.grid, 'res_x': RES, 'freq': MESH['freq']}
        self.model = utils.Model(**inp)

    def teardown(self, cycle):
        del self.grid, self.sfield, self.model

    def time_solver(self, cycle):
        solver.solver(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                cycle=cycle,
                semicoarsening=True,
                linerelaxation=True,
                verb=2)
