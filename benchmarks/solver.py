import numpy as np
from emg3d import utils, solver
from os.path import join, dirname


class Solver:
    """Timing for emg3d.solver.solver.

    See data/salt_create.py for the mesh- and data-creation.

    Currently, this fails for `sslsolver=False`. No idea why. If put into a
    `time_solver`-function instead of a class, it runs fine.

    """

    # Parameters to loop over
    params = [['iso', 'vti', 'tri']]
    param_names = ['anisotropy', ]

    def setup_cache(self):

        # Load data.
        DATA = np.load(join(dirname(__file__), 'data/salt_data.npz'),
                       allow_pickle=True)
        res = DATA['res'][()]
        mesh = DATA['mesh'][()]

        # Create grid.
        grid = utils.TensorMesh(
                [mesh['hx'], mesh['hy'], mesh['hz']], mesh['x0'])

        # Get source field.
        sfield = utils.get_source_field(grid, mesh['src'], mesh['freq'], 0)

        data = {
            'grid': grid,
            'sfield': sfield,
        }
        for anisotropy in self.params[0]:  # size

            # Create model.
            if anisotropy == 'iso':
                model = utils.Model(grid, res, freq=mesh['freq'])
            elif anisotropy == 'vti':
                model = utils.Model(grid, res, res_z=2*res, freq=mesh['freq'])
            else:
                model = utils.Model(grid, res, 2*res, 3*res, freq=mesh['freq'])

            data[anisotropy] = model

        # Run one iteration to ensure functions are jited.
        solver.solver(grid, model, sfield, sslsolver=True, maxit=1, verb=2)

        return data

    def time_solver_ssl(self, data, anisotropy):
        solver.solver(
                grid=data['grid'],
                model=data[anisotropy],
                sfield=data['sfield'],
                sslsolver=True,
                semicoarsening=True,
                linerelaxation=True,
                maxit=1,
                verb=2)

    def peakmem_solver_ssl(self, data, anisotropy):
        solver.solver(
                grid=data['grid'],
                model=data[anisotropy],
                sfield=data['sfield'],
                sslsolver=True,
                semicoarsening=True,
                linerelaxation=True,
                maxit=1,
                verb=2)
