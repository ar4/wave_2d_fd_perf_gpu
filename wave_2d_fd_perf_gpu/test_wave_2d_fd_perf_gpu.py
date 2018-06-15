"""Test the propagators."""
import pytest
import numpy as np
import scipy.special
from wave_2d_fd_perf_gpu.cuda import (VCuda1, VCuda2, VCuda3, VCuda4, VCuda5)
from wave_2d_fd_perf_gpu.numba import (VNumba1)
from wave_2d_fd_perf_gpu.pytorch import (VPytorch1, VPytorch2)
from wave_2d_fd_perf_gpu.pycuda import (VPycuda1)

def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency."""
    t = np.arange(length) * dt - peak_time
    y = ((1 - 2 * np.pi**2 * freq**2 * t**2)
         * np.exp(-np.pi**2 * freq**2 * t**2))
    return y.astype(np.float32)


def green(x, x_s, dx, dt, c, f):
    """Use the 2D Green's function to determine the wavefield at a given
    location and time due to the given source.
    """
    r = np.linalg.norm(x - x_s)
    nt = len(f)
    w = np.fft.rfftfreq(nt, dt)
    fw = np.fft.rfft(f)
    G = 1j / 4 * scipy.special.hankel1(0, -2 * np.pi * w * r / c)
    G[0] = 0
    s = G * fw * dx**2
    u = np.fft.irfft(s, nt)
    return u


@pytest.fixture
def model_one(c=1500, freq=25, dx=5, dt=0.0001, nx=[50, 50], batch_size=1,
              num_sources_per_shot=1, nsteps=None):
    """Create a constant model, and the expected wavefield."""
    model = np.ones(nx, dtype=np.float32) * c
    sz = (nx[0]/2 + np.arange(batch_size).reshape(-1, 1)
          + np.arange(num_sources_per_shot).reshape(1, -1)).astype(np.int32)
    sx = (nx[1]/2 + np.arange(batch_size).reshape(-1, 1)
          + np.arange(num_sources_per_shot).reshape(1, -1)).astype(np.int32)
    assert np.max(sz) < nx[0]
    assert np.max(sx) < nx[1]
    if nsteps is None:
        # time is chosen to avoid reflections from boundaries
        nsteps = int(0.9*(np.min([sz, sx, nx[0]-sz, nx[1]-sx]) * dx / c / dt))
    sources = np.tile(ricker(freq, nsteps, dt, 0.05), [batch_size,
                                                       num_sources_per_shot, 1])

    # direct wave
    expected = np.zeros([batch_size, nx[0], nx[1]], np.float32)
    for shotidx in range(batch_size):
        for sourceidx in range(num_sources_per_shot):
            for z in range(nx[0]):
                for x in range(nx[1]):
                    cell_loc = np.array([z, x])*dx
                    source_loc = np.array([sz[shotidx, sourceidx],
                                           sx[shotidx, sourceidx]])*dx
                    expected[shotidx, z, x] += \
                            green(cell_loc, source_loc, dx, dt, c,
                                  sources[shotidx, sourceidx])[-1]
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': sources, 'sz': sz, 'sx': sx, 'batch_size': batch_size,
            'expected': expected}


@pytest.fixture
def model_two(c=1500, dc=1000, freq=25, dx=5, dt=0.0001, nx=[20, 20],
              batch_size=1, num_sources_per_shot=1, nsteps=None,
              calc_expected=True):
    """Create a random model and compare with VCuda1 implementation."""
    np.random.seed(1)
    model = np.random.random(nx).astype(np.float32) * dc + c
    if nsteps is None:
        nsteps = int(0.2/dt)
    sz = np.random.randint(nx[0], size=[batch_size, num_sources_per_shot])
    sz = sz.astype(np.int32)
    sx = np.random.randint(nx[1], size=[batch_size, num_sources_per_shot])
    sx = sx.astype(np.int32)
    sources = np.zeros([batch_size, num_sources_per_shot, nsteps],
                       dtype=np.float32)
    for shotidx in range(batch_size):
        for sourceidx in range(num_sources_per_shot):
            peak_time = np.round((0.05+np.random.rand()*0.05)/dt)*dt
            sources[shotidx, sourceidx] = ricker(freq, nsteps, dt, peak_time)
    v = VCuda1(model, batch_size, dx, dt)
    expected = v.step(sources, sz, sx)
    v.finalise()
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': sources, 'sz': sz, 'sx': sx, 'batch_size': batch_size,
            'expected': expected}


@pytest.fixture
def versions():
    """Return a list of implementations."""
    return [VNumba1, VPytorch1, VPytorch2, VCuda1, VCuda2, VCuda3, VCuda4,
            VCuda5, VPycuda1]


def test_one_reflector(model_one, versions):
    """Verify that the numeric and analytic wavefields are similar."""

    for v in versions:
        _test_version(v, model_one, atol=1.5)


def test_allclose(model_two, versions):
    """Verify that all implementations produce similar results."""

    for v in versions[1:]:
        print(v.__name__)
        _test_version(v, model_two, atol=5e-4)


def _test_version(version, model, atol):
    """Run the test for one implementation."""
    v = version(model['model'], model['batch_size'], model['dx'], model['dt'])
    y = v.step(model['sources'], model['sz'], model['sx'])
    v.finalise()
    assert np.allclose(y, model['expected'], atol=atol)
