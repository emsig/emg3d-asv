"""
Create some example data.

See `emg3d-examples` => `2a_SEG-EAGE_3D-Salt-Model.ipynb`
"""
import discretize
import numpy as np
import scipy.interpolate as si

import emg3d

# => You have to provide the path to the SEG-EAGE salt model
PATH = '/home/dtr/Data/SEG-EAGE/3-D_Salt_Model/VEL_GRIDS/Saltf@@'


def get_orig_model():
    """Calculate resistivities from velocities."""

    # Dimensions
    nx, ny, nz = 676, 676, 210

    # Load data
    with open(PATH, 'r') as file:
        v = np.fromfile(file, dtype=np.dtype('float32').newbyteorder('>'))
        v = v.reshape(nx, ny, nz, order='F')

    # Velocity to resistivity transform for whole cube
    res = (v/1700)**3.88  # Sediment resistivity = 1

    # Overwrite basement resistivity from 3660 m onwards
    res[:, :, np.arange(nz)*20 > 3660] = 500.  # Resistivity of basement

    # Set sea-water to 0.3
    res[:, :, :15][v[:, :, :15] <= 1500] = 0.3

    # Fix salt resistivity
    res[v == 4482] = 30.

    # Flip z-axis
    res = np.flip(res, 2)

    # Create a discretize-mesh
    mesh = discretize.TensorMesh(
            [np.ones(nx)*20., np.ones(ny)*20., np.ones(nz)*20.],
            np.array([0, 0, -nz*20.]))

    return mesh, res


def create_smaller_model(mesh, res, size):
    """Create a smaller model of the original one."""

    src = [6400, 6600, 6500, 6500, -50, -50]  # source location
    freq = 1.0                                # Frequency

    # Get calculation domain as a function of frequency (resp., skin depth)
    if size == 'small':
        hx_min, xdomain = emg3d.utils.get_domain(
                x0=6500, freq=freq, limits=[2000, 11500], min_width=[5, 100])
        hz_min, zdomain = emg3d.utils.get_domain(
                freq=freq, limits=[-4180, 0], min_width=[5, 40], fact_pos=40)

        # Create stretched grid
        hx = emg3d.utils.get_stretched_h(hx_min, xdomain, 2**5, 6500)
        hy = emg3d.utils.get_stretched_h(hx_min, xdomain, 2**5, 6500)
        hz = emg3d.utils.get_stretched_h(hz_min, zdomain, 2**5, x0=-100, x1=0)
    else:
        hx_min, xdomain = emg3d.utils.get_domain(
                x0=6500, freq=freq, limits=[0, 13500], min_width=[5, 100])
        hz_min, zdomain = emg3d.utils.get_domain(
                freq=freq, limits=[-4180, 0], min_width=[5, 20], fact_pos=40)

        # Create stretched grid
        hx = emg3d.utils.get_stretched_h(hx_min, xdomain, 2**6, 6500)
        hy = emg3d.utils.get_stretched_h(hx_min, xdomain, 2**6, 6500)
        hz = emg3d.utils.get_stretched_h(hz_min, zdomain, 3*2**5, x0=-100,
                                         x1=0)

    grid = discretize.TensorMesh(
            [hx, hy, hz],
            np.array([xdomain[0], xdomain[0], zdomain[0]]))

    print(grid)

    fn = si.RegularGridInterpolator(
        (mesh.vectorCCx, mesh.vectorCCy, mesh.vectorCCz),
        res, bounds_error=False, fill_value=None)
    cres = fn(grid.gridCC, method='linear')

    # Create model
    model = emg3d.utils.Model(grid, cres, freq=freq)

    # Set air resistivity
    iz = np.argmin(np.abs(grid.vectorNz))
    model.res_x[:, :, iz:] = 2e14

    # Ensure at least top layer is water
    model.res_x[:, :, iz] = 0.3

    # Save resistivities and mesh
    save_mesh = {
        'hx': grid.hx,
        'hy': grid.hy,
        'hz': grid.hz,
        'x0': grid.x0,
        'src': src,
        'freq': freq,
        'res': model.res,
    }

    return save_mesh


# Create data
mesh, res = get_orig_model()
small = create_smaller_model(mesh, res, 'small')
big = create_smaller_model(mesh, res, 'big')

# Store data
np.savez_compressed('./salt_data.npz', small=small, big=big)
