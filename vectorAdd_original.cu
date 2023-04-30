/**
 * 2D convolution 
 *
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <mma.h>
// using namespace nvcuda;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

__global__ void conv1(const int *channelIn, const float *kernel, float *channelOut, int Ni, int Nn, int Nx, int Ny, int Kx, int Ky) 
{
  // Get current thread index
  int tIndex = blockDim.x * blockIdx.x + threadIdx.x;

  // stride 1
  int stride = 1;

  // TODO: Ensure we never go out of bounds

  // for (size_t i = 0; i < count; i++)
  // {
  //   /* code */
  // }
}


__global__ void wmma_example(half* a, half *b, float *c)
{
  // // Declare the fragments
  // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  // wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  // wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
  // wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  // // Load the inputs
  // wmma::load_matrix_sync(a_frag, a, 16);
  // wmma::load_matrix_sync(b_frag, b, 16);

  // // Perform the matrix multiplication
  // wmma::mma_sync(acc_frag, a_frag, b_frag, c_frag);

  // // Store the output
  // wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}


/**
 * Host main routine
 */
int main(void) 
{
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;


  // Input and output image sizes
  int Nx = 224;
  int Ny = 224;

  // Allocate memory for image (single channel)
  int **image;
  cudaMallocManaged(&image, Ny*sizeof(int));
  for (int i = 0; i < Ny; i++)
  {
    cudaMallocManaged(&image[i], Nx*sizeof(int));
  }
  
  // Initialize single channel image
  for (size_t x = 0; x < Nx; x++)
  {
    for (size_t y = 0; y < Ny; y++)
    {
      image[x][y] = 1;
    }
  }

  // Kernel size 
  int Kx = 3;
  int Ky = 3;

  // Allocate memory for kernel
  float **kernel;
  cudaMallocManaged(&kernel, Ky*sizeof(float));
  for (int i = 0; i < Ky; i++)
  {
    cudaMallocManaged(&kernel[i], Kx*sizeof(float));
  }
  // Initialize kernel
  for (size_t x = 0; x < Kx; x++)
  {
    for (size_t y = 0; y < Ky; y++)
    {
      kernel[x][y] = 0.5;
    }
  }

  // Output feature map size
  int Ni = 64;
  int Nn = 64;

  float **featureMap;
  cudaMallocManaged(&featureMap, Nn*sizeof(float));
  for (int i = 0; i < Nn; i++)
  {
    cudaMallocManaged(&featureMap[i], Ni*sizeof(float));
  }





  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  float *h_A = (float *)malloc(size);

  // Allocate the host input vector B
  float *h_B = (float *)malloc(size);

  // Allocate the host output vector C
  float *h_C = (float *)malloc(size);

  // Verify that allocations succeeded
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate the device input vector A
  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = cudaFree(d_A);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done\n");
  return 0;
}
