/**
 * CS 251 - Mini Project 1
 * Convolutional Neural Network with CUDA
 *
 * Author: Ryan Dougherty 
 */

#include <stdio.h>
#include <algorithm>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <mma.h>

using namespace std;
using namespace nvcuda;

/*
  * ReLU activation function
*/
__device__ float relu(float x)
{
  return x > 0 ? x : 0;
}


// Tiling contstants. 
#define Tn  32
#define Ti  32

#define Ty  8
#define Tx  8

// ==================== HELPER FUNCTIONS ====================
/*
 * Helper function to get a command line argument
*/
char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

/*
 * Helper function to check if a command line argument exists
*/
bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

/*
 * Print our a matrix from rows and colums
 */
void printMatrix(float *matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    printf("[ ");
    for (int j = 0; j < cols; j++)
    {
      printf("%f ", matrix[i * cols + j]);
    }
    printf(" ]\n");
  }
}

/*
 * Generate a random float between -0.5 and 0.5
 */
float randFloat()
{
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
}

/*
 * Fill the weights with random values
 */
template<int Ky, int Kx, int Nn, int Ni, int Ny, int Nx>
void fillWeights(float (*weights)[Ky][Kx][Nn][Ni])
{
  for (size_t ky = 0; ky < Ky; ky++)
    for (size_t kx = 0; kx < Kx; kx++)
      for (size_t nn = 0; nn < Nn; nn++)
        for (size_t ni = 0; ni < Ni; ni++)
          (*weights)[ky][kx][nn][ni] = randFloat();
}

/*
 * Fill the image with random values
 */
template<int Ky, int Kx, int Nn, int Ni, int Ny, int Nx>
void fillImage(float (*image)[Ny + Ky][Nx + Kx][Ni])
{
  for (size_t y = 0; y < Ny + Ky; y++)
    for (size_t x = 0; x < Nx + Kx; x++)
      for (size_t ni = 0; ni < Ni; ni++)
        (*image)[y][x][ni] = randFloat();
}

/*
* Fill the synapse with random values
*/
template<int Nn, int Ni>
void fillSynapse(float (*synapse)[Nn][Ni])
{
  for (size_t n = 0; n < Nn; n++)
    for (size_t i = 0; i < Ni; i++)
      (*synapse)[n][i] = randFloat();
}

/*
* Fill the input with random values
*/
template<int Ni>
void fillInput(float (*input)[Ni])
{
  for (size_t i = 0; i < Ni; i++)
    (*input)[i] = randFloat();
}

// ==================== CONVOLUTION FUNCTIONS ====================
/*
 * Convolution 1 function
*/
template<int Ky, int Kx, int Nn, int Ni, int Ny, int Nx>
__global__ void conv1(const float (*weights)[Ky][Kx][Nn][Ni], 
                      const float (*input)[Ny + Ky][Nx + Kx][Ni],
                      float (*output)[Ny][Nx][Nn])
{
  // Get current thread index
  int tIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float sum[Nn] = {0};

  // Stride over the input feature map
  for (int t = tIndex; t < Ny * Nx; t += stride)
  {
    // Get our y and x indices-Ti+1
    int y = t / Nx;
    int x = t % Nx;

    // Tiling over the kernel
    for (int nn = 0; nn < Nn-Tn+1; nn += Tn)
    {
      // Initialize sum to 0 for this tiling
      for (int n = nn; n < nn + Tn; n++)
        sum[n] = 0;

      for (int ky = 0; ky < Ky; ky++)
        for (int kx = 0; kx < Kx; kx++)
          for (int ii = 0; ii < Ni; ii += Ti)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = ii; i < ii + Ti; i++)
              {
                float curWeight = (*weights)[ky][kx][n][i];
                float curInput = (*input)[y + ky][x + kx][i];
                sum[n] += curWeight * curInput;
              }

      // Apply ReLU activation function and store output
      for (int n = nn; n < nn + Tn; n++)
        (*output)[y][x][n] = relu(sum[n]);
    }
  }
}

/*
 * Convolution 2 function
 * 
 *  This is an adjusted version of Conv1 that loops around the feature maps.
 *  This is particulatly efficient when we have a large number of feature maps.
*/
template<int Ky, int Kx, int Nn, int Ni, int Ny, int Nx>
__global__ void conv2(const float (*weights)[Ky][Kx][Nn][Ni], 
                      const float (*input)[Ny + Ky][Nx + Kx][Ni],
                      float (*output)[Ny][Nx][Nn])
{

  // Get current thread index
  int tIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Stride over the input feature map
  for (int t = tIndex; t < Nn * Ni; t += stride)
  {
    // Get our n and i indices of our feature map
    int n = t / Nn;
    int i = t % Nn;

    // Loop over weights and tile over the input
    for (int ky = 0; ky < Ky; ky++)
      for (int kx = 0; kx < Kx; kx++)
        for (int yy = 0; yy < Ny; yy += Ty)
          for (int xx = 0; xx < Nx; xx += Tx )
            for (int y = yy; y < yy + Ty; y++)
              for (int x = xx; x < xx + Tx; x++)
              {
                float curWeight = (*weights)[ky][kx][n][i];
                float curInput = (*input)[y + ky][x + kx][i];
                (*output)[y][x][n] += relu(curWeight * curInput);
              }
  }
}

// ==================== CLASSIFIER FUNCTION ====================
/*
 * Classifier function
*/
template<int Nn, int Ni>
__global__ void classifier(const float (*synapse)[Nn][Ni], 
                          const float (*input)[Ni],
                          float (*output)[Nn])
{
  int tIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Stride threads over the synapse dimension
  for (int n = tIndex; n < Nn; n += stride)
  {
    float sum = 0;
    
    // Tile over input dimension
    for (int ii = 0; ii < Ni; ii+=Ti)
      for (int i = ii; i < ii + Ti; i++)
        sum += (*synapse)[n][i] * (*input)[i];

    (*output)[n] = relu(sum);
  }
}


// ==================== RUNNER FUNCTIONS ====================
/*
* runConv - Helper function to allocate memory and run a convolution
*/
template<int Ky, int Kx, int Nn, int Ni, int Ny, int Nx>
void runConv(int batchSize, string convName)
{
  int blockSize = 1024;
  int numBlocks = (Ny*Nx + blockSize - 1) / blockSize;

  // For Conv2, use a number of blocks equal to the number of feature maps instead of the number of pixels
  // (since we're looping over the feature maps)
  if(convName == "Conv2")
    numBlocks = (Nn*Ni + blockSize - 1) / blockSize;

  printf("Starting convolution with BlockSize=%i, NumBlocks=%i\n", blockSize, numBlocks);

  // Allocate memory for kernel weights
  float (*weights)[Ky][Kx][Nn][Ni];
  cudaMallocManaged(&weights, (Ky * Kx * Ni * Nn) * sizeof(float));
  fillWeights<Ky, Kx, Nn, Ni, Ky, Kx>(weights);

  // Allocate memory for image 
  float (*image)[Ny + Ky][Nx + Kx][Ni];
  cudaMallocManaged(&image, ((Ny + Ky) * (Nx + Kx) * Ni) * sizeof(float));

  // Allocate memory for the output feature map. No need to initialize, since it's the output
  float (*output)[Ny][Nx][Nn];
  cudaMallocManaged(&output, (Ny * Nx * Nn)*sizeof(float));

  for (int i = 0; i < batchSize; i++)
  {
    printf("Starting Convolution [Batch %i of %i] (Ky=%i, Kx=%i, Nn=%i, Ni=%i, Ny=%i, Nx=%i)...\n", i, batchSize, Ky, Kx, Nn, Ni, Ny, Nx);

    fillImage<Ky, Kx, Nn, Ni, Ny, Nx>(image);

    // Compute convolution
    if(convName == "Conv1")
    {
      conv1<Ky, Kx, Nn, Ni, Ny, Nx><<<numBlocks, blockSize>>>(weights, image, output);
    }
    else
    {
      conv2<Ky, Kx, Nn, Ni, Ny, Nx><<<numBlocks, blockSize>>>(weights, image, output);
    }
  }
  printf("Finished Convolution! (Ky=%i, Kx=%i, Nn=%i, Ni=%i, Ny=%i, Nx=%i)...\n", Ky, Kx, Nn, Ni, Ny, Nx);

  // Wait for GPU to finish
  cudaDeviceSynchronize();

  // Print out an output 
  printMatrix((float*)output, 100, 100);

  // Free our memory
  cudaFree(image);
  cudaFree(weights);
  cudaFree(output);
}

/*
 * runClasifier - Helper function to allocate memory and run a classifier
*/
template<int Nn, int Ni>
void runClasifier(int batchSize)
{
  int blockSize = 1024;
  int numBlocks = (Ni*Nn + blockSize - 1) / blockSize;

  float (*synapse)[Nn][Ni];
  cudaMallocManaged(&synapse, (Nn*Ni) * sizeof(float));
  fillSynapse<Nn, Ni>(synapse);

  // Define input and output
  float(*input)[Ni];
  cudaMallocManaged(&input, (Ni) * sizeof(float));

  float(*output)[Nn];
  cudaMallocManaged(&output, (Nn) * sizeof(float));

  for (int i = 0; i < batchSize; i++)
  {
    printf("Starting Classifier [Batch %i of %i] (Nn=%i, Ni=%i)...\n", i, batchSize, Nn, Ni);

    fillInput<Ni>(input);

    classifier<Nn, Ni><<<numBlocks, blockSize>>>(synapse, input, output);

  }
  printf("Finished Classifier! (Nn=%i, Ni=%i)...\n", Nn, Ni);

  cudaDeviceSynchronize();

  // Print out an output 
  printMatrix((float*)output, 10, 1);

  // Free memory
  cudaFree(synapse);
  cudaFree(input);
  cudaFree(output);
}


/*
 * Example of using the cuda core matrix multiply functions
 * NOTE: Never used for this project, but left here for reference
*/
__global__ void wmma_example(half* a, half *b, float *c)
{
  // Declare the fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  // Load the inputs
  wmma::load_matrix_sync(a_frag, a, 16);
  wmma::load_matrix_sync(b_frag, b, 16);

  // Perform the matrix multiplication
  wmma::mma_sync(acc_frag, a_frag, b_frag, c_frag);

  // Store the output
  wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}


/**
 * Host main routine
 */
int main(int argc, char *argv[]) 
{
  printf("Starting...\n");


  // Get the kernel to run. Default to Conv1
  string kernel = "Conv1";
  if(cmdOptionExists(argv, argv+argc, "-k"))
  {
    kernel = getCmdOption(argv, argv+argc, "-k");
  }
  else if(cmdOptionExists(argv, argv+argc, "--kernel"))
  {
    kernel = getCmdOption(argv, argv+argc, "--kernel");
  }

  // Assign our number of batches
  int batches = 1;
  if(cmdOptionExists(argv, argv+argc, "-b"))
  {
    batches = stoi(getCmdOption(argv, argv+argc, "-b"));
  }
  else if(cmdOptionExists(argv, argv+argc, "--batches"))
  {
    batches = stoi(getCmdOption(argv, argv+argc, "--batches"));
  }

  // Run our given kernel
  if(kernel == "Conv1")
  {
    printf("Running Conv1...\n");
    runConv<3, 3, 64, 64, 224, 224>(batches, kernel);
  }
  else if(kernel == "Conv2")
  {
    printf("Running Conv2...\n");
    // Run our Conv2
    runConv<3, 3, 512, 512, 14, 14>(batches, kernel);
  }
  else if(kernel == "Classifier1")
  {
    printf("Running Classifier1\n");
    runClasifier<25088, 4096>(batches);
  }
  else if(kernel == "Classifier2")
  {
    printf("Running Classifier2\n");
    runClasifier<4096, 1024>(batches);
  }
  else
  {
    printf("Unknown kernel %s\n", kernel.c_str());
    return 1;
  }

  printf("Done\n");
  return 0;
}
