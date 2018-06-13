/* Two kernels, no shared memory, manual laplacian, 2D malloc */
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                bool abort=true)
{
        if (code != cudaSuccess)
        {
                fprintf(stderr, "GPUassert: %s %s %d\n",
                                cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

#define M(a, z, x) (*((float *) ((char *)a + z * model_pitch) + x))
#define WF(a, b, z, x) (*((float *) ((char *)a + b * nz * model_pitch + (z) * model_pitch) + x))
#define SA(a, b, s, t) (*((float *) ((char *)a + b * ns * source_amplitude_pitch + s * source_amplitude_pitch) + t))
#define SL(a, b, s) (*((int *) ((char *)a + b * sources_loc_pitch) + s))

__constant__ float fd_d[3];
size_t model_pitch_h;

// Device code
__global__ void step_d(const float *const model,
                float *wfc,
                float *wfp,
                const int nb, const int nz, const int nx,
                const size_t model_pitch)
{
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int z = blockDim.y * blockIdx.y + threadIdx.y;
        int b = blockDim.z * blockIdx.z + threadIdx.z;
        float lap;
        bool in_domain = (x > 1) && (x < nx - 2)
                && (z > 1) && (z < nz - 2)
                && (b < nb);

        if (in_domain)
        {
                /* Laplacian */
                lap = (fd_d[0] * WF(wfc, b, z, x) +
                                fd_d[1] *
                                (WF(wfc, b, z, x + 1) +
                                 WF(wfc, b, z, x - 1) +
                                 WF(wfc, b, z + 1, x) +
                                 WF(wfc, b, z - 1, x)) +
                                fd_d[2] *
                                (WF(wfc, b, z, x + 2) +
                                 WF(wfc, b, z, x - 2) +
                                 WF(wfc, b, z + 2, x) +
                                 WF(wfc, b, z - 2, x)));

                /* Main evolution equation */
                WF(wfp, b, z, x) = M(model, z, x) * lap + 2 * WF(wfc, b, z, x)
                        - WF(wfp, b, z, x);

        }
}

__global__ void add_sources_d(const float *const model,
                float *wfp,
                const float *const source_amplitude,
                const int *const sources_z,
                const int *const sources_x,
                const int nz, const int nx,
                const int nt, const int ns, const int it,
                const size_t model_pitch, const size_t source_amplitude_pitch,
                const size_t sources_loc_pitch)
{

        int x = threadIdx.x;
        int b = blockIdx.x;
        int sz = SL(sources_z, b, x);
        int sx = SL(sources_x, b, x);
        WF(wfp, b, sz, sx) += SA(source_amplitude, b, x, it) * M(model, sz, sx);
}

// Host code

        extern "C"
void setup(int nb, int nz, int nx, float dx, float *model_h,
                float **model_d, float **wfc_d, float **wfp_d)
{
        float fd[3] = {
                -10.0f / 2 / (dx * dx),
                4.0f / 3 / (dx * dx),
                -1.0f / 12 / (dx * dx)
        };
        gpuErrchk(cudaMemcpyToSymbol(fd_d, fd, 3*sizeof(float)));

        gpuErrchk(cudaMallocPitch(model_d, &model_pitch_h, nx * sizeof(float),
                                nz));
        gpuErrchk(cudaMemcpy2D(*model_d, model_pitch_h, model_h,
                                nx * sizeof(float), nx * sizeof(float),
                                nz, cudaMemcpyHostToDevice));

        gpuErrchk(cudaMallocPitch(wfc_d, &model_pitch_h, nx * sizeof(float),
                                nb * nz));
        gpuErrchk(cudaMemset2D(*wfc_d, model_pitch_h, 0, nx * sizeof(float),
                                nb * nz));

        gpuErrchk(cudaMallocPitch(wfp_d, &model_pitch_h, nx * sizeof(float),
                                nb * nz));
        gpuErrchk(cudaMemset2D(*wfp_d, model_pitch_h, 0, nx * sizeof(float),
                                nb * nz));
}

extern "C"
void step(int nb, int nz, int nx, int nt, int ns,
                float *model_d, float *wfc_d, float *wfp_d,
                float *source_amplitude_h,
                int *sources_z_h, int *sources_x_h, float *wfc_h)
{

        size_t source_amplitude_pitch;
        size_t sources_loc_pitch;

        float *source_amplitude_d;
        gpuErrchk(cudaMallocPitch(&source_amplitude_d,
                                &source_amplitude_pitch,
                                nt * sizeof(float), nb * ns));
        gpuErrchk(cudaMemcpy2D(source_amplitude_d, source_amplitude_pitch,
                                source_amplitude_h,
                                nt * sizeof(float), nt * sizeof(float),
                                nb * ns, cudaMemcpyHostToDevice));

        int *sources_z_d;
        gpuErrchk(cudaMallocPitch(&sources_z_d, &sources_loc_pitch,
                                ns * sizeof(int), nb));
        gpuErrchk(cudaMemcpy2D(sources_z_d, sources_loc_pitch, sources_z_h,
                                ns * sizeof(int), ns * sizeof(int),
                                nb, cudaMemcpyHostToDevice));


        int *sources_x_d;
        gpuErrchk(cudaMallocPitch(&sources_x_d, &sources_loc_pitch,
                                ns * sizeof(int), nb));
        gpuErrchk(cudaMemcpy2D(sources_x_d, sources_loc_pitch, sources_x_h,
                                ns * sizeof(int), ns * sizeof(int),
                                nb, cudaMemcpyHostToDevice));

        dim3 dimBlock(32, 32, 1);
        int gridx = (nx + dimBlock.x - 1) / dimBlock.x;
        int gridz = (nz + dimBlock.y - 1) / dimBlock.y;
        int gridb = (nb + dimBlock.z - 1) / dimBlock.z;
        dim3 dimGrid(gridx, gridz, gridb);

        int it;
        float *tmp;

        for (it = 0; it < nt; it++)
        {
                step_d<<<dimGrid, dimBlock>>>(model_d, wfc_d, wfp_d,
                                nb, nz, nx, model_pitch_h);
                gpuErrchk( cudaPeekAtLastError() );
                add_sources_d<<<nb, ns>>>(model_d, wfp_d,
                                source_amplitude_d, sources_z_d, sources_x_d,
                                nz, nx, nt, ns, it, model_pitch_h,
                                source_amplitude_pitch, sources_loc_pitch);
                gpuErrchk( cudaPeekAtLastError() );

                tmp = wfc_d;
                wfc_d = wfp_d;
                wfp_d = tmp;
        }

        gpuErrchk(cudaMemcpy2D(wfc_h, nx * sizeof(float), wfc_d,
                                model_pitch_h, nx * sizeof(float),
                                nb * nz, cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(source_amplitude_d));
        gpuErrchk(cudaFree(sources_z_d));
        gpuErrchk(cudaFree(sources_x_d));

}

        extern "C"
void finalise(float *model_d, float *wfc_d, float *wfp_d)
{
        gpuErrchk(cudaFree(model_d));
        gpuErrchk(cudaFree(wfc_d));
        gpuErrchk(cudaFree(wfp_d));
}
