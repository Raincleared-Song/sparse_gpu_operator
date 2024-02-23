#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

// Row Major
// (32, 32, 1) (mat_row / 32)
__global__ void ffn_4(nv_bfloat16 *mat, nv_bfloat16 *vec, nv_bfloat16 *res,
                      unsigned int mat_row, unsigned int mat_col)
{

    float sum = 0;
    // nv_bfloat16 sum = __float2bfloat16(0.0f);
    __shared__ float warp_sum[32];
    warp_sum[threadIdx.x] = 0.0f;

    unsigned int col_id = blockIdx.y * 32 + threadIdx.x; // (0,512) (0,32), max:32*511+32=16384
    nv_bfloat16 *res_p = &res[col_id];
    unsigned int warp_id = threadIdx.y; // (0,32)
    unsigned int row_id = warp_id;
    nv_bfloat16 *vec_p = &vec[row_id];
    nv_bfloat16 *mat_p = &mat[row_id * mat_col + col_id];
    nv_bfloat16 mat_val = __float2bfloat16(0.0f);
#pragma unroll 32
    for (int iter = 0; iter < mat_row; iter = iter + 32)
    {
        nv_bfloat16 vec_val = vec_p[iter];
        if (__bfloat162float(vec_val) == 0.0f)
            continue;
        else
            mat_val = mat_p[iter * mat_col];
        sum += __bfloat162float(vec_val) * __bfloat162float(mat_val);
    }
    atomicAdd(&warp_sum[threadIdx.x], sum);

    __syncthreads();
    if (warp_id == 0)
    {
        // Write final result
        float sum = warp_sum[threadIdx.x];
        *res_p = __float2bfloat16(sum);
    }
}

void launch_ffn_4(nv_bfloat16 *mat, nv_bfloat16 *vec, nv_bfloat16 *res,
                  unsigned int mat_row, unsigned int mat_col)
{
    mat_row = 11008;
    mat_col = 4096;

    dim3 grid_dim(1, mat_col / 32);
    dim3 block_dim(32, 32, 1);

    ffn_4<<<grid_dim, block_dim>>>(mat, vec, res, mat_row, mat_col);

}