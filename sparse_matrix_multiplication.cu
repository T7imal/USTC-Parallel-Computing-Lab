#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstddef>
#include <cstdlib>
#include <chrono>

__global__ void spmm(int *dense, int *sparse, int *result, size_t pitch, int M, int N, int P, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < P)
    {
        // 将需要的数据读取进共享内存
        // extern __shared__ int dense_row_and_sparse_col[];
        int sum = 0;
        int *sparse_col = (int *)((char *)sparse + col * pitch);
        for (int i = 0; i < sparse_col[0]; i++)
        {
            sum += dense[row * N + sparse_col[i * 2 + 1]] * sparse_col[i * 2 + 2];
        }
        result[row * P + col] = sum;
    }
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    int M, N, P, K;
    scanf("%d %d %d %d", &M, &N, &P, &K);
    int *dense = (int *)malloc(M * N * sizeof(int));
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            scanf("%d", &dense[i * N + j]);
        }
    }

    // 按列存储稀疏矩阵，每一列的元素存储在同一个数组中，数组的第一个元素代表该列有多少个非零元素
    // 后面的元素依次存储非零元素的行号和值
    int *sparse = (int *)malloc(P * (2 * K + 1) * sizeof(int));
    for (int i = 0; i < P; i++)
    {
        sparse[i * (2 * K + 1)] = 0;
    }
    for (int i = 0; i < K; i++)
    {
        int row, col, val;
        scanf("%d %d %d", &row, &col, &val);
        sparse[col * (2 * K + 1)]++;
        sparse[col * (2 * K + 1) + sparse[col * (2 * K + 1)] * 2 - 1] = row;
        sparse[col * (2 * K + 1) + sparse[col * (2 * K + 1)] * 2] = val;
    }

    int *result = (int *)malloc(M * P * sizeof(int));

    int *d_dense, *d_sparse, *d_result;
    cudaMalloc(&d_dense, M * N * sizeof(int));
    size_t pitch;
    cudaMallocPitch((void **)&d_sparse, &pitch, (2 * K + 1) * sizeof(int), P);
    cudaMalloc(&d_result, M * P * sizeof(int));

    cudaMemcpy(d_dense, dense, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_sparse, pitch, sparse, (2 * K + 1) * sizeof(int), (2 * K + 1) * sizeof(int), P,
                 cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (P + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 共享内存的长度为(N+2K+1)*sizeof(int)
    spmm<<<numBlocks, threadsPerBlock>>>(d_dense, d_sparse, d_result, pitch, M, N, P, K);

    cudaMemcpy(result, d_result, M * P * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_dense);
    cudaFree(d_sparse);
    cudaFree(d_result);

    auto end = std::chrono::high_resolution_clock::now();

    // printf("Time: %f ms\n", std::chrono::duration<double, std::milli>(end - start).count());

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < P; j++)
        {
            printf("%d ", result[i * P + j]);
        }
        printf("\n");
    }
    return 0;
}