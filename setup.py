#!/usr/bin/env python
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('wave_2d_fd_perf_gpu', parent_package, top_path)
    config.add_subpackage('wave_2d_fd_perf_gpu')
    config.add_data_files('Makefile')
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
