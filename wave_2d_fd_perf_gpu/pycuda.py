import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from wave_2d_fd_perf_gpu.propagators import Propagator


class VPycuda(Propagator):
    """PyCUDA implementations."""
    def __init__(self, jitfunc1, jitfunc2, fd_d, model, batch_size, pad, dx,
                 dt=None):
        super(VPycuda, self).__init__(model, batch_size, pad, dx, dt)
        self.jitfunc1 = jitfunc1
        self.jitfunc2 = jitfunc2

        # allocate and copy model to GPU
        self.model_d = drv.mem_alloc(self.model_pad.nbytes)
        drv.memcpy_htod(self.model_d, self.model_pad)

        # allocate wavefields
        self.wfc = np.zeros([batch_size, self.nz_padded, self.nx_padded],
                            np.float32)
        self.wfc_d = drv.mem_alloc(self.wfc.nbytes)
        self.wfp_d = drv.mem_alloc(self.wfc.nbytes)

        # create and copy finite difference coeffs to constant memory
        self.fd_d = fd_d
        fd = np.array([-10/2, 4/3, -1/12], np.float32) / dx**2
        drv.memcpy_htod(self.fd_d, fd)

        # set block and grid dimensions
        threadsperblockx = 32
        blockspergridx = ((self.nx_padded + (threadsperblockx - 1))
                          // threadsperblockx)
        threadsperblockz = 32
        blockspergridz = ((self.nz_padded + (threadsperblockz - 1))
                          // threadsperblockz) * self.batch_size

        self.griddim = blockspergridx, blockspergridz
        self.blockdim = threadsperblockx, threadsperblockz, 1

    def step(self, source_amplitude, sources_z, sources_x):

        ns = source_amplitude.shape[1]
        nt = source_amplitude.shape[2]

        for it in range(nt):
            self.jitfunc1(self.model_d, self.wfc_d, self.wfp_d,
                          np.int32(self.batch_size),
                          np.int32(self.nz_padded), np.int32(self.nx_padded),
                          grid=self.griddim, block=self.blockdim)
            self.jitfunc2(self.model_d, self.wfp_d,
                          drv.In(source_amplitude),
                          drv.In(sources_z + self.pad),
                          drv.In(sources_x + self.pad),
                          np.int32(self.nz_padded), np.int32(self.nx_padded),
                          np.int32(nt), np.int32(ns), np.int32(it),
                          grid=(self.batch_size, 1), block=(ns, 1, 1))
            self.wfc_d, self.wfp_d = self.wfp_d, self.wfc_d

        drv.memcpy_dtoh(self.wfc, self.wfc_d)

        return self.wfc[:, self.pad:-self.pad, self.pad:-self.pad]

    def finalise(self):
        del (self.model_d, self.wfc, self.wfc_d, self.wfp_d,
             self.jitfunc1, self.jitfunc2)


class VPycuda1(VPycuda):
    def __init__(self, model, batch_size, dx, dt=None):

        source = """
__constant__ float fd_d[3];

__global__ void step_d(const float *const model,
                float *wfc,
                float *wfp,
                const int nb, const int nz, const int nx)
{
        int zblocks_per_shot = gridDim.y / nb;
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int z = blockDim.y * (blockIdx.y % zblocks_per_shot) + threadIdx.y;
        int b = blockIdx.y / zblocks_per_shot;
        int i = z * nx + x;
        int ib = b * nz * nx + i;
        float lap;
        bool in_domain = (x > 1) && (x < nx - 2)
                && (z > 1) && (z < nz - 2)
                && (b < nb);

        if (in_domain)
        {
                /* Laplacian */
                lap = (fd_d[0] * wfc[ib] +
                                fd_d[1] *
                                (wfc[ib + 1] +
                                 wfc[ib - 1] +
                                 wfc[ib + nx] +
                                 wfc[ib - nx]) +
                                fd_d[2] *
                                (wfc[ib + 2] +
                                 wfc[ib - 2] +
                                 wfc[ib + 2 * nx] +
                                 wfc[ib - 2 * nx]));

                /* Main evolution equation */
                wfp[ib] = model[i] * lap + 2 * wfc[ib] - wfp[ib];

        }
}

__global__ void add_sources_d(const float *const model,
                float *wfp,
                const float *const source_amplitude,
                const int *const sources_z,
                const int *const sources_x,
                const int nz, const int nx,
                const int nt, const int ns, const int it)
{

        int x = threadIdx.x;
        int b = blockIdx.x;
        int i = sources_z[b * ns + x] * nx + sources_x[b * ns + x];
        int ib = b * nz * nx + i;
        wfp[ib] += source_amplitude[b * ns * nt + x * nt + it] * model[i];
}
"""

        mod = SourceModule(source,
                           options=['-ccbin', 'clang-3.8', '--restrict',
                                    '--use_fast_math', '-O3'])
        jitfunc1 = mod.get_function('step_d')
        jitfunc2 = mod.get_function('add_sources_d')
        fd_d = mod.get_global('fd_d')[0]

        pad = 3
        super(VPycuda1, self).__init__(jitfunc1, jitfunc2, fd_d, model,
                                       batch_size, pad, dx, dt)
