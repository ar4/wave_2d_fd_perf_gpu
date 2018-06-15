"""Measure the runtime of the propagators."""
from timeit import repeat
import itertools
import numpy as np
import pandas as pd
from wave_2d_fd_perf_gpu.cuda import (VCuda1, VCuda2, VCuda3, VCuda4, VCuda5)
from wave_2d_fd_perf_gpu.numba import (VNumba1)
from wave_2d_fd_perf_gpu.pytorch import (VPytorch1, VPytorch2)
from wave_2d_fd_perf_gpu.pycuda import (VPycuda1)
from wave_2d_fd_perf_gpu.test_wave_2d_fd_perf_gpu import (ricker, model_two)

def run_timing_num_steps(num_repeat=10, model_size=1000,
                         batch_size=10, num_sources_per_shot=1,
                         num_steps=2**np.arange(14),
                         versions=None):
    """Time implementations as num_steps varies."""
    return run_timing(num_repeat, ([model_size], [batch_size],
                                   [num_sources_per_shot],
                                   num_steps), versions)


def run_timing_model_size(num_repeat=10, model_sizes=range(200, 2200, 200),
                          batch_size=10, num_sources_per_shot=1, num_step=10,
                          versions=None):
    """Time implementations as model size varies."""
    return run_timing(num_repeat, (model_sizes, [batch_size],
                                   [num_sources_per_shot],
                                   [num_step]), versions)

def run_timing(num_repeat, configs, versions=None):
    """Time implementations."""

    if versions is None:
        versions = _versions()

    config_str = ['model_size', 'batch_size', 'num_sources_per_shot',
                  'num_step']

    times = pd.DataFrame(columns=['version', 'time'] + config_str)

    for config in itertools.product(*configs):
        config_dict = dict(zip(config_str, config))
        print(config_dict)
        model = _make_model(**config_dict)
        times = _time_versions(versions, model, config_dict, num_repeat,
                               times)
    return times


def _versions():
    """Return a list of versions to be timed."""
    return [{'class': VNumba1, 'name': 'Numba1'},
            {'class': VCuda1, 'name': 'Cuda1'},
            {'class': VCuda2, 'name': 'Cuda2'},
            {'class': VCuda3, 'name': 'Cuda3'},
            {'class': VCuda4, 'name': 'Cuda4'},
            {'class': VCuda5, 'name': 'Cuda5'},
            {'class': VPycuda1, 'name': 'Pycuda1'},
            {'class': VPytorch1, 'name': 'Pytorch1'},
            {'class': VPytorch2, 'name': 'Pytorch2'}]


def _make_model(model_size, batch_size, num_sources_per_shot, num_step):
    """Create a model with a given number of elements and time steps."""
    return model_two(nx=[model_size, model_size], batch_size=batch_size,
                     num_sources_per_shot=num_sources_per_shot,
                     nsteps=num_step, calc_expected=False)


def _time_versions(versions, model, config_dict, num_repeat, dataframe):
    """Loop over versions and append the timing results to the dataframe."""
    for v in versions:
        time = _time_version(v['class'], model, num_repeat)
        version_dict = config_dict.copy()
        version_dict.update({'version': v['name'], 'time': time})
        dataframe = dataframe.append(version_dict, ignore_index=True)
    return dataframe


def _time_version(version, model, num_repeat):
    """Time a particular version."""
    v = version(model['model'], model['batch_size'], model['dx'], model['dt'])

    def closure():
        """Closure over variables so they can be used in repeat below."""
        v.step(model['sources'], model['sz'], model['sx'])

    time = np.min(repeat(closure, number=1, repeat=num_repeat))
    v.finalise();
    return time

if __name__ == '__main__':
    print(run_timing_num_steps())
