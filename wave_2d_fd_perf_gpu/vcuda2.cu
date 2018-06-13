/* Two kernels, no shared memory, CUDNN laplacian, 1D malloc */

/* I used Peter Goldsborough's description of how to use CuDNN:
   http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/ */
#include <stdio.h>
#include <cudnn.h>

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
#define cudnnErrchk(ans) { cudnnAssert((ans), __FILE__, __LINE__); }
inline void cudnnAssert(cudnnStatus_t code, const char *file, int line,
                bool abort=true)
{
        if (code != CUDNN_STATUS_SUCCESS)
        {
                fprintf(stderr, "CUDNNassert: %s %s %d\n",
                                cudnnGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

static cudnnHandle_t cudnn;
static cudnnTensorDescriptor_t input_descriptor;
static cudnnTensorDescriptor_t output_descriptor;
static cudnnFilterDescriptor_t kernel_descriptor;
static cudnnConvolutionDescriptor_t convolution_descriptor;
static cudnnConvolutionFwdAlgo_t convolution_algorithm;
static size_t workspace_bytes;
static void *workspace_d;
static float *lap_d;
static float *kernel_d;

// Device code
__global__ void step_d(const float *const model,
                float *wfc,
                float *wfp,
                float *lap,
                const int nb, const int nz, const int nx)
{
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int z = blockDim.y * blockIdx.y + threadIdx.y;
        int b = blockDim.z * blockIdx.z + threadIdx.z;
        int i = z * nx + x;
        int ib = b * nz * nx + i;
        bool in_domain = (x < nx) && (z < nz) && (b < nb);

        if (in_domain)
        {
                /* Main evolution equation */
                wfp[ib] = model[i] * lap[ib] + 2 * wfc[ib] - wfp[ib];

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
        float kernel[25] = {
                0.0f, 0.0f, -1.0f / 12 / (dx * dx), 0.0f, 0.0f,
                0.0f, 0.0f, 4.0f / 3 / (dx * dx), 0.0f, 0.0f,
                -1.0f / 12 / (dx * dx),
                4.0f / 3 / (dx * dx),
                -10.0f / 2 / (dx * dx),
                4.0f / 3 / (dx * dx),
                -1.0f / 12 / (dx * dx),
                0.0f, 0.0f, 4.0f / 3 / (dx * dx), 0.0f, 0.0f,
                0.0f, 0.0f, -1.0f / 12 / (dx * dx), 0.0f, 0.0f
        };

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

        /* CuDNN setup */
        cudnnErrchk(cudnnCreate(&cudnn));
        cudnnErrchk(cudnnCreateTensorDescriptor(&input_descriptor));
        cudnnErrchk(cudnnSetTensor4dDescriptor(input_descriptor,
                                /*format=*/CUDNN_TENSOR_NCHW,
                                /*dataType=*/CUDNN_DATA_FLOAT,
                                /*batch_size=*/nb,
                                /*channels=*/1,
                                /*image_height=*/nz,
                                /*image_width=*/nx));

        //cudnnTensorDescriptor_t output_descriptor;
        cudnnErrchk(cudnnCreateTensorDescriptor(&output_descriptor));
        cudnnErrchk(cudnnSetTensor4dDescriptor(output_descriptor,
                                 /*format=*/CUDNN_TENSOR_NCHW,
                                 /*dataType=*/CUDNN_DATA_FLOAT,
                                 /*batch_size=*/nb,
                                 /*channels=*/1,
                                 /*image_height=*/nz,
                                 /*image_width=*/nx));

        //cudnnFilterDescriptor_t kernel_descriptor;
        cudnnErrchk(cudnnCreateFilterDescriptor(&kernel_descriptor));
        cudnnErrchk(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                 /*dataType=*/CUDNN_DATA_FLOAT,
                                 /*format=*/CUDNN_TENSOR_NCHW,
                                 /*out_channels=*/1,
                                 /*in_channels=*/1,
                                 /*kernel_height=*/5,
                                 /*kernel_width=*/5));

        //cudnnConvolutionDescriptor_t convolution_descriptor;
        cudnnErrchk(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        cudnnErrchk(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                 /*pad_height=*/2,
                                 /*pad_width=*/2,
                                 /*vertical_stride=*/1,
                                 /*horizontal_stride=*/1,
                                 /*dilation_height=*/1,
                                 /*dilation_width=*/1,
                                 /*mode=*/CUDNN_CROSS_CORRELATION,
                                 /*computeType=*/CUDNN_DATA_FLOAT));

        /* Instead of using the heuristic 'Get' method, I will use the 'Find'
           method for choosing the algorithm. This requires that I allocate
           some temporary workspace. I arbitrarily choose to allocate 10 times
           the space required to store the wavefield. After the algorithm has
           been chosen, I deallocate this space and then allocate the correct
           amount of workspace for the chosen algorithm. */

        //cudnnConvolutionFwdAlgo_t convolution_algorithm;
//        cudnnErrchk(cudnnGetConvolutionForwardAlgorithm(cudnn,
//                                 input_descriptor,
//                                 kernel_descriptor,
//                                 convolution_descriptor,
//                                 output_descriptor,
//                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//                                 /*memoryLimitInBytes=*/0,
//                                 &convolution_algorithm));


        workspace_bytes = nwfbytes * 10;
        gpuErrchk(cudaMalloc(&workspace_d, workspace_bytes));
        gpuErrchk(cudaMalloc(&lap_d, nwfbytes));
        gpuErrchk(cudaMalloc(&kernel_d, 5 * 5 * sizeof(float)));
        gpuErrchk(cudaMemcpy(kernel_d, kernel, 5 * 5 * sizeof(float),
                                cudaMemcpyHostToDevice));

        int algoCount;
        cudnnConvolutionFwdAlgoPerf_t perfResults;
	cudnnErrchk(cudnnFindConvolutionForwardAlgorithmEx(cudnn,
                                input_descriptor,
                                *wfc_d,
                                kernel_descriptor,
                                kernel_d,
                                convolution_descriptor,
                                output_descriptor,
                                lap_d,
                                1,
                                &algoCount,
                                &perfResults,
                                workspace_d,
                                workspace_bytes));
        convolution_algorithm = perfResults.algo;

        cudnnErrchk(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                input_descriptor,
                                kernel_descriptor,
                                convolution_descriptor,
                                output_descriptor,
                                convolution_algorithm,
                                &workspace_bytes));

        gpuErrchk(cudaFree(workspace_d));
        gpuErrchk(cudaMalloc(&workspace_d, workspace_bytes));

}

extern "C"
void check(int *batch_size, int *channels, int *height, int *width, size_t *wb)
{

        cudnnErrchk(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   batch_size,
                                                   channels,
                                                   height,
                                                   width));
        *wb = workspace_bytes;
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

        dim3 dimBlock(32, 32, 1);
        int gridx = (nx + dimBlock.x - 1) / dimBlock.x;
        int gridz = (nz + dimBlock.y - 1) / dimBlock.y;
        int gridb = (nb + dimBlock.z - 1) / dimBlock.z;
        dim3 dimGrid(gridx, gridz, gridb);

        int it;
        float *tmp;
        const float alpha = 1.0f, beta = 0.0f;

        for (it = 0; it < nt; it++)
        {
                
                cudnnErrchk(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     wfc_d,
                                     kernel_descriptor,
                                     kernel_d,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     workspace_d,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     lap_d));

                step_d<<<dimGrid, dimBlock>>>(model_d, wfc_d, wfp_d, lap_d,
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
        gpuErrchk(cudaFree(workspace_d));
        gpuErrchk(cudaFree(lap_d));
        gpuErrchk(cudaFree(kernel_d));
        cudnnErrchk(cudnnDestroyTensorDescriptor(input_descriptor));
        cudnnErrchk(cudnnDestroyTensorDescriptor(output_descriptor));
        cudnnErrchk(cudnnDestroyFilterDescriptor(kernel_descriptor));
        cudnnErrchk(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
        
        cudnnErrchk(cudnnDestroy(cudnn));
}
