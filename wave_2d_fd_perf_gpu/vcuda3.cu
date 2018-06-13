/* Two kernels, shared memory, manual laplacian, 1D malloc */
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

#define BLOCKSIZEZ 32
#define BLOCKSIZEX 32
#define PAD 3
#define SHAREDH (BLOCKSIZEZ + 2 * PAD)
#define SHAREDW (BLOCKSIZEX + 2 * PAD)
#define INDOMAIN ((b < nb) && (z + shiftz > 0) && (z + shiftz < nz) && (x + shiftx > 0) && (x + shiftx < nx))
#define SHIFTSHARED (shiftz * SHAREDW + shiftx)
#define SHIFTWF (shiftz * (nx) + shiftx)

__constant__ float fd_d[3];

// Device code
__global__ void step_d(const float *const model,
                float *wfc,
                float *wfp,
                const int nb, const int nz, const int nx)
{
        __shared__ float temp[SHAREDH * SHAREDW];
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int z = blockDim.y * blockIdx.y + threadIdx.y;
        int b = blockDim.z * blockIdx.z + threadIdx.z;
        int im = z * nx + x;
        int iwf = b * nz * nx + im;
        int ishared = (threadIdx.y + PAD) * SHAREDW + threadIdx.x + PAD;
        int shiftz;
        int shiftx;
        float lap;

        /* top left */
        shiftz = -PAD;
        shiftx = -PAD;
        if INDOMAIN
                temp[ishared + SHIFTSHARED] = wfc[iwf + SHIFTWF];
        else
                temp[ishared + SHIFTSHARED] = 0.0f;

        /* right edge */
        if (threadIdx.x < 2 * PAD)
        {
                shiftz = 0;
                shiftx = BLOCKSIZEX - PAD;
                if INDOMAIN
                        temp[ishared + SHIFTSHARED] = wfc[iwf + SHIFTWF];
                else
                        temp[ishared + SHIFTSHARED] = 0.0f;
        }

        /* bottom edge */
        if (threadIdx.y < 2 * PAD)
        {
                shiftz = BLOCKSIZEZ - PAD;
                shiftx = 0;
                if INDOMAIN
                        temp[ishared + SHIFTSHARED] = wfc[iwf + SHIFTWF];
                else
                        temp[ishared + SHIFTSHARED] = 0.0f;
        }

        /* bottom right corner */
        if ((threadIdx.x >= blockDim.x - 2 * PAD) &&
                        (threadIdx.y >= blockDim.y - 2 * PAD))
        {
                shiftz = PAD;
                shiftx = PAD;
                if INDOMAIN
                        temp[ishared + SHIFTSHARED] = wfc[iwf + SHIFTWF];
                else
                        temp[ishared + SHIFTSHARED] = 0.0f;
        }

        __syncthreads();

        shiftz = 0;
        shiftx = 0;
        if INDOMAIN
        {
                /* Laplacian */
                lap = (fd_d[0] * temp[ishared] +
                                fd_d[1] *
                                (temp[ishared + 1] +
                                 temp[ishared - 1] +
                                 temp[ishared + SHAREDW] +
                                 temp[ishared - SHAREDW]) +
                                fd_d[2] *
                                (temp[ishared + 2] +
                                 temp[ishared - 2] +
                                 temp[ishared + 2 * SHAREDW] +
                                 temp[ishared - 2 * SHAREDW]));

                /* Main evolution equation */
                wfp[iwf] = model[im] * lap + 2 * wfc[iwf] - wfp[iwf];
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

        int nmodel = nz * nx;
        int nwf = nb * nmodel;
        size_t nmodelbytes = nmodel * sizeof(float);
        size_t nwfbytes = nwf * sizeof(float);

        gpuErrchk(cudaMalloc(model_d, nmodelbytes));
        gpuErrchk(cudaMemcpy(*model_d, model_h, nmodelbytes,
                                cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc(wfc_d, nwfbytes));
        gpuErrchk(cudaMemset(*wfc_d, 0, nwfbytes));

        gpuErrchk(cudaMalloc(wfp_d, nwfbytes));
        gpuErrchk(cudaMemset(*wfc_d, 0, nwfbytes));

}

extern "C"
void step(int nb, int nz, int nx, int nt, int ns,
                float *model_d, float *wfc_d, float *wfp_d,
                float *source_amplitude_h,
                int *sources_z_h, int *sources_x_h, float *wfc_h)
{

        int tns = nb * ns;

        float *source_amplitude_d;
        size_t nbytes = tns * nt * sizeof(float);
        gpuErrchk(cudaMalloc(&source_amplitude_d, nbytes));
        gpuErrchk(cudaMemcpy(source_amplitude_d, source_amplitude_h, nbytes,
                                cudaMemcpyHostToDevice));

        int *sources_z_d;
        nbytes = tns * sizeof(int);
        gpuErrchk(cudaMalloc(&sources_z_d, nbytes));
        gpuErrchk(cudaMemcpy(sources_z_d, sources_z_h, nbytes,
                                cudaMemcpyHostToDevice));

        int *sources_x_d;
        nbytes = tns * sizeof(int);
        gpuErrchk(cudaMalloc(&sources_x_d, nbytes));
        gpuErrchk(cudaMemcpy(sources_x_d, sources_x_h, nbytes,
                                cudaMemcpyHostToDevice));


        dim3 dimBlock(BLOCKSIZEX, BLOCKSIZEZ, 1);
        int gridx = (nx + dimBlock.x - 1) / dimBlock.x;
        int gridz = (nz + dimBlock.y - 1) / dimBlock.y;
        int gridb = (nb + dimBlock.z - 1) / dimBlock.z;
        dim3 dimGrid(gridx, gridz, gridb);

        int it;
        float *tmp;

        for (it = 0; it < nt; it++)
        {
                step_d<<<dimGrid, dimBlock>>>(model_d, wfc_d, wfp_d,
                                nb, nz, nx);
                gpuErrchk( cudaPeekAtLastError() );
                add_sources_d<<<nb, ns>>>(model_d, wfp_d,
                                source_amplitude_d, sources_z_d, sources_x_d,
                                nz, nx, nt, ns, it);
                gpuErrchk( cudaPeekAtLastError() );

                tmp = wfc_d;
                wfc_d = wfp_d;
                wfp_d = tmp;
        }

        int nwf = nb * nz * nx;
        size_t nwfbytes = nwf * sizeof(float);
        gpuErrchk(cudaMemcpy(wfc_h, wfc_d, nwfbytes, cudaMemcpyDeviceToHost));

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
