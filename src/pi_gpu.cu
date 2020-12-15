#include "check.h"
#include <math.h>
#include "timing.h"

/**
  * This reduction kernel adds multiple elements per thread sequentially, and then the threads
  * work together to produce a block sum in shared memory.  The code is optimized using
  * warp-synchronous programming to eliminate unnecessary barrier synchronization. Performing
  * sequential work in each thread before performing the log(N) parallel summation reduces the
  * overall cost of the algorithm while keeping the work complexity O(n) and the step complexity 
  * O(log n). (Brent's Theorem optimization)
*/
template <unsigned int blockSize, bool nIsPow2>
__global__ void pi_gpu_kernel(double *sum, double h, unsigned long n)
{
    if (blockSize == 1)
    {
        if ((n == 1) || (n == 2))
        {
            const double x = (0 + 0.5) * h;
            *sum = 4.0 / (1.0 + x * x);
        }
        if (n == 2)
        {
            const double x = (1 + 0.5) * h;    
            *sum += 4.0 / (1.0 + x * x);
        }
    }
    else
    {
        // volatile is needed to prevent the last (tid < 32) reduction
        // step from incorrect overoptimization
        volatile extern __shared__ double shsum[];

        // perform first level of reduction,
        // reading from global memory, writing to shared memory
        unsigned long tid = threadIdx.x;
        unsigned long i = tid + blockIdx.x * (blockSize * 2);
        unsigned int gridSize = blockSize * 2 * gridDim.x;
        double mySum = 0.0;

        // we reduce multiple elements per thread.  The number is determined by the 
        // number of active thread blocks (via gridDim).  More blocks will result
        // in a larger gridSize and therefore fewer elements per thread
        while (i < n)
        {
            const double x = (i + 0.5) * h;
            mySum += 4.0 / (1.0 + x * x);

            // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
            if (nIsPow2 || i + blockSize < n)
            {
                const double x = (i + blockSize + 0.5) * h;
                mySum += 4.0 / (1.0 + x * x);
            }

            i += gridSize;
        } 

        shsum[tid] = mySum;
        __syncthreads();

        // do reduction in shared mem
        if (blockSize >= 512) { if (tid < 256) { shsum[tid] = mySum += shsum[tid + 256]; } __syncthreads(); }
        if (blockSize >= 256) { if (tid < 128) { shsum[tid] = mySum += shsum[tid + 128]; } __syncthreads(); }
        if (blockSize >= 128) { if (tid <  64) { shsum[tid] = mySum += shsum[tid +  64]; } __syncthreads(); }
    
        if (tid < 32)
        {
            if (blockSize >=  64) { shsum[tid] = mySum += shsum[tid + 32]; }
            if (blockSize >=  32) { shsum[tid] = mySum += shsum[tid + 16]; }
            if (blockSize >=  16) { shsum[tid] = mySum += shsum[tid +  8]; }
            if (blockSize >=   8) { shsum[tid] = mySum += shsum[tid +  4]; }
            if (blockSize >=   4) { shsum[tid] = mySum += shsum[tid +  2]; }
            if (blockSize >=   2) { shsum[tid] = mySum += shsum[tid +  1]; }
        }

        // write result for this block to global mem 
        if (tid == 0) 
            atomicAdd(sum, shsum[0]);
    }   
}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		printf("Usage: %s <n>\n", argv[0]);
		return 0;
	}

	unsigned long n = strtoul(argv[1], NULL, 0);
	printf("n = %lu\n", n);

	double start;
	get_time(&start);

        cudaDeviceProp props;
        CUDA_ERR_CHECK(cudaGetDeviceProperties(&props, 0));
	constexpr int szblock = 128;
	int nblocksPerSM = props.maxThreadsPerMultiProcessor / szblock;
        size_t szshmemPerBlock = std::min(props.sharedMemPerBlock, props.sharedMemPerMultiprocessor / nblocksPerSM);

	double h = 1.0 / n;

	double pi;
	double* pi_gpu = NULL;
	CUDA_ERR_CHECK(cudaMalloc(&pi_gpu, sizeof(double)));
	CUDA_ERR_CHECK(cudaMemset(pi_gpu, 0, sizeof(double)));
	unsigned int nblocks = n / szblock;
	if (n % szblock) nblocks++;
	pi_gpu_kernel<szblock, false><<<nblocks, szblock, szshmemPerBlock>>>(pi_gpu, h, n);
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
	CUDA_ERR_CHECK(cudaMemcpy(&pi, pi_gpu, sizeof(double), cudaMemcpyDeviceToHost));

	double finish;
        get_time(&finish);

	pi *= h;
	printf("pi = %f, err = %e\n", pi, abs(pi - M_PI));
	printf("time = %f sec\n", finish - start);

	return 0;
}

