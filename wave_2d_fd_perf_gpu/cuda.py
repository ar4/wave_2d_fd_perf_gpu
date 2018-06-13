from ctypes import c_int, c_float, POINTER, byref, c_size_t
import numpy as np
import wave_2d_fd_perf_gpu
from wave_2d_fd_perf_gpu.propagators import Propagator

class VCuda(Propagator):
    """CUDA implementations."""
    def __init__(self, libname, model_h, batch_size, pad, dx, dt=None):
        super(VCuda, self).__init__(model_h, batch_size, pad, dx, dt)

        self._libvc = np.ctypeslib.load_library(libname,
                                                wave_2d_fd_perf_gpu.__path__[0])

        self._libvc.setup.argtypes = \
                [c_int, # batch_size
                 c_int, # nz_padded
                 c_int, # nx_padded
                 c_float, # dx
                 np.ctypeslib.ndpointer(dtype=c_float, ndim=2,
                                        shape=(self.nz_padded,
                                               self.nx_padded),
                                        flags=('C_CONTIGUOUS')), # model_h
                 POINTER(POINTER(c_float)), # model_d
                 POINTER(POINTER(c_float)), # wfc_d
                 POINTER(POINTER(c_float))] # wfp_d

        self._libvc.step.argtypes = \
                [c_int, # batch_size
                 c_int, # nz_padded
                 c_int, # nx_padded
                 c_int, # num_steps
                 c_int, # ns
                 POINTER(c_float), # model_d
                 POINTER(c_float), # wfc_d
                 POINTER(c_float), # wfp_d
                 np.ctypeslib.ndpointer(dtype=c_float, ndim=3,
                                        #shape=(nb * ns * nt),
                                        flags=('C_CONTIGUOUS')), # source_amp_h
                 np.ctypeslib.ndpointer(dtype=c_int, ndim=2,
                                        #shape=(nb * ns),
                                        flags=('C_CONTIGUOUS')), # sources_z_h
                 np.ctypeslib.ndpointer(dtype=c_int, ndim=2,
                                        #shape=(nb * ns),
                                        flags=('C_CONTIGUOUS')), # sources_x_h
                 np.ctypeslib.ndpointer(dtype=c_float, ndim=3,
                                        shape=(self.batch_size,
                                               self.nz_padded,
                                               self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE'))]
                                        # wfp_h

        self._libvc.finalise.argtypes = \
                [POINTER(c_float), # model_d
                 POINTER(c_float), # wfc_d
                 POINTER(c_float)] # wfp_d

        self.model_pad_d = POINTER(c_float)()
        self.wfc_d = POINTER(c_float)()
        self.wfp_d = POINTER(c_float)()

        self._libvc.setup(self.batch_size, self.nz_padded, self.nx_padded,
                          dx, self.model_pad,
                          byref(self.model_pad_d), byref(self.wfc_d),
                          byref(self.wfp_d))

    def step(self, source_amplitude_h, sources_z_h, sources_x_h):

        wfc_h = np.zeros([self.batch_size,
                          self.nz_padded,
                          self.nx_padded], np.float32)

        ns = source_amplitude_h.shape[1]
        num_steps = source_amplitude_h.shape[2]

        self._libvc.step(self.batch_size, self.nz_padded, self.nx_padded,
                         num_steps, ns, self.model_pad_d, self.wfc_d,
                         self.wfp_d, source_amplitude_h,
                         sources_z_h + self.pad, sources_x_h + self.pad, wfc_h)

        if num_steps % 2 == 1:
            self.wfc_d, self.wfp_d = self.wfp_d, self.wfc_d

        return wfc_h[:, self.pad:self.nz_padded-self.pad,
                     self.pad:self.nx_padded-self.pad]

    def finalise(self):
        self._libvc.finalise(self.model_pad_d, self.wfc_d, self.wfp_d)
        del self.model_pad_d, self.wfc_d, self.wfp_d, self._libvc


class VCuda1(VCuda):
    def __init__(self, model_h, batch_size, dx, dt=None):
        libname = 'libvcuda1'
        pad = 3
        super(VCuda1, self).__init__(libname, model_h, batch_size, pad, dx, dt)


class VCuda2(VCuda):
    def __init__(self, model_h, batch_size, dx, dt=None):
        libname = 'libvcuda2'
        pad = 0
        super(VCuda2, self).__init__(libname, model_h, batch_size, pad, dx, dt)

    def check(self):
        self._libvc.check.argtypes = \
                [POINTER(c_int), # batch_size
                 POINTER(c_int), # chen
                 POINTER(c_int), # h
                 POINTER(c_int), # w
                 POINTER(c_size_t)] # wb
        bs = c_int(0)
        chan = c_int(0)
        h = c_int(0)
        w = c_int(0)
        wb = c_size_t(0)
        self._libvc.check(byref(bs), byref(chan), byref(h), byref(w), byref(wb))
        print(bs, chan, h, w, wb)


class VCuda3(VCuda):
    def __init__(self, model_h, batch_size, dx, dt=None):
        libname = 'libvcuda3'
        pad = 0
        super(VCuda3, self).__init__(libname, model_h, batch_size, pad, dx, dt)


class VCuda4(VCuda):
    def __init__(self, model_h, batch_size, dx, dt=None):
        libname = 'libvcuda4'
        pad = 3
        super(VCuda4, self).__init__(libname, model_h, batch_size, pad, dx, dt)


class VCuda5(VCuda):
    def __init__(self, model_h, batch_size, dx, dt=None):
        libname = 'libvcuda5'
        pad = 3
        super(VCuda5, self).__init__(libname, model_h, batch_size, pad, dx, dt)
