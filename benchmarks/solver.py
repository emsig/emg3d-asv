import numpy as np
from emg3d import utils, solver
from os.path import join, dirname


class Solver:
    """Timing for emg3d.solver.solver.

    See data/salt_create.py for the mesh- and data-creation.

    Move it to functions:

    - One for memory
      - with/without sslsolver
      - iso/vti/tri

    - Timing, with/without sslsolver; semicoarsening and line relaxation

    - Timing, MG with/without semicoarsening; with/without line relaxation

    - Timing, MG all cycles; semicoarsening and line relaxation

    """

    # Parameters to loop over
    params = [[True, False], ['iso', 'vti', 'tri'], ]
    param_names = ['sslsolver', 'anisotropy', ]

    def setup(self, sslsolver, anisotropy):

        # Load data.
        DATA = np.load(join(dirname(__file__), 'data/salt_data.npz'),
                       allow_pickle=True)
        res = DATA['res'][()]
        mesh = DATA['mesh'][()]

        # Create grid.
        self.grid = utils.TensorMesh(
                [mesh['hx'], mesh['hy'], mesh['hz']], mesh['x0'])

        # Get source field.
        self.sfield = utils.get_source_field(
                self.grid, mesh['src'], mesh['freq'], 0)

        # Create model.
        inp = {'grid': self.grid, 'res_x': res, 'freq': mesh['freq']}
        if anisotropy == 'iso':
            self.model = utils.Model(**inp)
        elif anisotropy == 'vti':
            self.model = utils.Model(res_z=3*res, **inp)
        else:
            self.model = utils.Model(res_y=2*res, res_z=3*res, **inp)

        # Delete unused variables.
        del DATA, res, mesh

    def teardown(self, sslsolver, anisotropy):
        del self.grid, self.sfield, self.model

    def time_solver(self, sslsolver, anisotropy):
        solver.solver(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                sslsolver=sslsolver,
                semicoarsening=True,
                linerelaxation=True,
                verb=2)

    def peakmem_solver(self, sslsolver, anisotropy):
        solver.solver(
                grid=self.grid,
                model=self.model,
                sfield=self.sfield,
                sslsolver=sslsolver,
                semicoarsening=True,
                linerelaxation=True,
                verb=2)
