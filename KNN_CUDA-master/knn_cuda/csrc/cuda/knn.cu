/** Modifed version of knn-CUDA from https://github.com/vincentfpgarcia/kNN-CUDA
 * Last modified by Christopher B. Choy <chrischoy@ai.stanford.edu> 12/23/2016
 */

#include <cstdio>
#include <stdint.h> // Added for int64_t support
#include "cuda.h"

#define BLOCK_DIM 16
#define DEBUG 0

__global__ void cuComputeDistanceGlobal(float* A, int wA, float* B, int wB, int dim, float* AB){
  __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
  __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];
  __shared__ int begin_A;
  __shared__ int begin_B;
  __shared__ int step_A;
  __shared__ int step_B;
  __shared__ int end_A;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  float tmp;
  float ssd = 0;

  begin_A = BLOCK_DIM * blockIdx.y;
  begin_B = BLOCK_DIM * blockIdx.x;
  step_A  = BLOCK_DIM * wA;
  step_B  = BLOCK_DIM * wB;
  end_A   = begin_A + (dim-1) * wA;

  int cond0 = (begin_A + tx < wA);
  int cond1 = (begin_B + tx < wB);
  int cond2 = (begin_A + ty < wA);

  for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
    if (a/wA + ty < dim){
      shared_A[ty][tx] = (cond0)? A[a + wA * ty + tx] : 0;
      shared_B[ty][tx] = (cond1)? B[b + wB * ty + tx] : 0;
    }
    else{
      shared_A[ty][tx] = 0;
      shared_B[ty][tx] = 0;
    }
    __syncthreads();

    if (cond2 && cond1){
      for (int k = 0; k < BLOCK_DIM; ++k){
        tmp = shared_A[k][ty] - shared_B[k][tx];
        ssd += tmp*tmp;
      }
    }
    __syncthreads();
  }

  if (cond2 && cond1)
    AB[(begin_A + ty) * wB + begin_B + tx] = ssd;
}

// CHANGED: long* to int64_t*
__global__ void cuInsertionSort(float *dist, int64_t *ind, int width, int height, int k){
  int l, i, j;
  float *p_dist;
  int64_t *p_ind; // CHANGED
  float curr_dist, max_dist;
  int64_t curr_row, max_row; // CHANGED
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (xIndex<width){
    p_dist   = dist + xIndex;
    p_ind    = ind  + xIndex;
    max_dist = p_dist[0];
    p_ind[0] = 1;

    for (l=1; l<k; l++){
      curr_row  = (int64_t)l * width; // CHANGED
      curr_dist = p_dist[curr_row];
      if (curr_dist<max_dist){
        i=l-1;
        for (int a=0; a<l-1; a++){
          if (p_dist[a*width]>curr_dist){
            i=a;
            break;
          }
        }
        for (j=l; j>i; j--){
          p_dist[j*width] = p_dist[(j-1)*width];
          p_ind[j*width]   = p_ind[(j-1)*width];
        }
        p_dist[i*width] = curr_dist;
        p_ind[i*width]  = l + 1;
      } else {
        p_ind[l*width] = l + 1;
      }
      max_dist = p_dist[curr_row];
    }

    max_row = (int64_t)(k-1)*width; // CHANGED
    for (l=k; l<height; l++){
      curr_dist = p_dist[l*width];
      if (curr_dist<max_dist){
        i=k-1;
        for (int a=0; a<k-1; a++){
          if (p_dist[a*width]>curr_dist){
            i=a;
            break;
          }
        }
        for (j=k-1; j>i; j--){
          p_dist[j*width] = p_dist[(j-1)*width];
          p_ind[j*width]   = p_ind[(j-1)*width];
        }
        p_dist[i*width] = curr_dist;
        p_ind[i*width]   = l + 1;
        max_dist             = p_dist[max_row];
      }
    }
  }
}

__global__ void cuParallelSqrt(float *dist, int width, int k){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  if (xIndex<width && yIndex<k)
    dist[yIndex*width + xIndex] = sqrt(dist[yIndex*width + xIndex]);
}

// CHANGED: long* to int64_t*
void debug(float * dist_dev, int64_t * ind_dev, const int query_nb, const int k){
  float* dist_host = new float[query_nb * k];
  int64_t* idx_host  = new int64_t[query_nb * k]; // CHANGED

  cudaMemcpy(dist_host, dist_dev, query_nb * k * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(idx_host, ind_dev, query_nb * k * sizeof(int64_t), cudaMemcpyDeviceToHost); // CHANGED

  int i, j;
  for(i = 0; i < k; i++){
    for (j = 0; j < query_nb; j++) {
      if (j % 8 == 0) printf("/\n");
      printf("%f ", sqrt(dist_host[i*query_nb + j]));
    }
    printf("\n");
  }
}

// CHANGED: long* to int64_t*
void knn_device(float* ref_dev, int ref_nb, float* query_dev, int query_nb,
    int dim, int k, float* dist_dev, int64_t* ind_dev, cudaStream_t stream){

  dim3 g_16x16(query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, 1);
  dim3 t_16x16(BLOCK_DIM, BLOCK_DIM, 1);
  if (query_nb % BLOCK_DIM != 0) g_16x16.x += 1;
  if (ref_nb   % BLOCK_DIM != 0) g_16x16.y += 1;

  dim3 g_256x1(query_nb / 256, 1, 1);
  dim3 t_256x1(256, 1, 1);
  if (query_nb%256 != 0) g_256x1.x += 1;

  dim3 g_k_16x16(query_nb / BLOCK_DIM, k / BLOCK_DIM, 1);
  dim3 t_k_16x16(BLOCK_DIM, BLOCK_DIM, 1);
  if (query_nb % BLOCK_DIM != 0) g_k_16x16.x += 1;
  if (k  % BLOCK_DIM != 0) g_k_16x16.y += 1;

  cuComputeDistanceGlobal<<<g_16x16, t_16x16, 0, stream>>>(ref_dev, ref_nb, query_dev, query_nb, dim, dist_dev);

  cuInsertionSort<<<g_256x1, t_256x1, 0, stream>>>(dist_dev, ind_dev, query_nb, ref_nb, k);

  cuParallelSqrt<<<g_k_16x16,t_k_16x16, 0, stream>>>(dist_dev, query_nb, k);
}