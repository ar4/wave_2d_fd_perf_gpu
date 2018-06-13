from numba import cuda
import numpy as np
from wave_2d_fd_perf_gpu.propagators import Propagator


class VNumba(Propagator):
    """Numba implementations."""
    def __init__(self, jitfunc1, jitfunc2, model, batch_size, pad, dx, dt=None):
        super(VNumba, self).__init__(model, batch_size, pad, dx, dt)
        self.wfc = np.zeros([batch_size, self.nz_padded, self.nx_padded],
                            np.float32)
        self.wfp = np.zeros_like(self.wfc)
        self.fd = np.array([-10/2, 4/3, -1/12], np.float32) / dx**2
        self.jitfunc1 = jitfunc1
        self.jitfunc2 = jitfunc2

    def step(self, source_amplitude, sources_z, sources_x):
        threadsperblockx = 32
        blockspergridx = ((self.nx_padded + (threadsperblockx - 1))
                          // threadsperblockx)
        threadsperblockz = 16
        blockspergridz = ((self.nz_padded + (threadsperblockz - 1))
                          // threadsperblockz)
        threadsperblockb = 1
        blockspergridb = ((self.batch_size + (threadsperblockb - 1))
                          // threadsperblockb)

        griddim = blockspergridx, blockspergridz, blockspergridb
        blockdim = threadsperblockx, threadsperblockz, threadsperblockb

        ns = source_amplitude.shape[1]
        nt = source_amplitude.shape[2]

        for it in range(nt):
            self.jitfunc1[griddim, blockdim](self.model_pad, self.wfc, self.wfp,
                                            self.fd, self.batch_size,
                                            self.nz_padded, self.nx_padded)
            self.jitfunc2[self.batch_size, ns](self.model_pad, self.wfp,
                                               source_amplitude,
                                               sources_z + self.pad,
                                               sources_x + self.pad, it)
            self.wfc, self.wfp = self.wfp, self.wfc

        return self.wfc[:, self.pad:-self.pad, self.pad:-self.pad]

    def finalise(self):
        del self.wfc, self.wfp, self.fd, self.jitfunc1, self.jitfunc2


class VNumba1(VNumba):
    def __init__(self, model, batch_size, dx, dt=None):

        @cuda.jit('void(float32[:, :], float32[:, :, :], float32[:, :, :], float32[:], int32, int32, int32)')
        def jitfunc1(model, wfc, wfp, fd, nb, nz, nx):
            x, z, b = cuda.grid(3)
            in_domain = ((z > 1) and (z < nz - 2) and
                         (x > 1) and (x < nx - 2) and
                         (b < nb))
            if in_domain:
                lap = (fd[0] * wfc[b, z, x] +
                       fd[1] *
                       (wfc[b, z, x + 1] +
                        wfc[b, z, x - 1] +
                        wfc[b, z + 1, x] +
                        wfc[b, z - 1, x]) +
                       fd[2] *
                       (wfc[b, z, x + 2] +
                        wfc[b, z, x - 2] +
                        wfc[b, z + 2, x] +
                        wfc[b, z - 2, x]))
                wfp[b, z, x] = (model[z, x] * lap
                                + 2 * wfc[b, z, x] - wfp[b, z, x])

        @cuda.jit('void(float32[:, :], float32[:, :, :], float32[:, :, :], int32[:, :], int32[:, :], int32)')
        def jitfunc2(model, wfp, source_amplitude, sources_z, sources_x, i):
            j = cuda.threadIdx.x
            b = cuda.blockIdx.x
            sz = sources_z[b, j]
            sx = sources_x[b, j]
            wfp[b, sz, sx] += source_amplitude[b, j, i] * model[sz, sx]

        pad = 3
        super(VNumba1, self).__init__(jitfunc1, jitfunc2, model, batch_size,
                                      pad, dx, dt)
