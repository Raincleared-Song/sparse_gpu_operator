#include <cuda.h>
// #include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

// Col major
__global__ void ffn_fuse_23(nv_bfloat16 *vec_sparse, nv_bfloat16 *vec_input,
                            nv_bfloat16 *mat_up, nv_bfloat16 *res, unsigned int mat_row,
                            unsigned int mat_col, float threshold)
{
    int col_id = blockIdx.y * 32 + threadIdx.y;
    int num_per_threadx = mat_row / 32;
    int row_chunk_id = threadIdx.x;
    int row_id = row_chunk_id * num_per_threadx;

    nv_bfloat16 *vec_sparse_p = &vec_sparse[col_id];            // per thread
    nv_bfloat16 *vec_input_p = &vec_input[row_id];              // per thread
    nv_bfloat16 *mat_up_p = &mat_up[col_id * mat_row + row_id]; // per thread, col-major
    nv_bfloat16 *res_p = &res[col_id];                          // per thread

    float4 *vec_input_f4 = reinterpret_cast<float4 *>(vec_input_p);
    float4 vec_input_f_val;
    float4 *mat_up_f4 = reinterpret_cast<float4 *>(mat_up_p);
    float4 mat_up_f_val;

    float sum = 0;
    nv_bfloat16 vec_sparse_val = *vec_sparse_p;
    if (__bfloat162float(vec_sparse_val) <= threshold)
        ;
    else
    {
#pragma unroll
        for (int i = 0; i < (num_per_threadx / 8) /*8个half*/; i++)
        {
            vec_input_f_val = vec_input_f4[i];
            const nv_bfloat162 *vec_input_h1 = (nv_bfloat162 *)&vec_input_f_val.x;
            const nv_bfloat162 *vec_input_h2 = (nv_bfloat162 *)&vec_input_f_val.y;
            const nv_bfloat162 *vec_input_h3 = (nv_bfloat162 *)&vec_input_f_val.z;
            const nv_bfloat162 *vec_input_h4 = (nv_bfloat162 *)&vec_input_f_val.w;

            mat_up_f_val = mat_up_f4[i];
            const nv_bfloat162 *mat_up_h1 = (nv_bfloat162 *)&mat_up_f_val.x;
            const nv_bfloat162 *mat_up_h2 = (nv_bfloat162 *)&mat_up_f_val.y;
            const nv_bfloat162 *mat_up_h3 = (nv_bfloat162 *)&mat_up_f_val.z;
            const nv_bfloat162 *mat_up_h4 = (nv_bfloat162 *)&mat_up_f_val.w;

            sum += __bfloat162float(vec_input_h1->x) * __bfloat162float(mat_up_h1->x);
            sum += __bfloat162float(vec_input_h1->y) * __bfloat162float(mat_up_h1->y);
            sum += __bfloat162float(vec_input_h2->x) * __bfloat162float(mat_up_h2->x);
            sum += __bfloat162float(vec_input_h2->y) * __bfloat162float(mat_up_h2->y);
            sum += __bfloat162float(vec_input_h3->x) * __bfloat162float(mat_up_h3->x);
            sum += __bfloat162float(vec_input_h3->y) * __bfloat162float(mat_up_h3->y);
            sum += __bfloat162float(vec_input_h4->x) * __bfloat162float(mat_up_h4->x);
            sum += __bfloat162float(vec_input_h4->y) * __bfloat162float(mat_up_h4->y);
        }
    }

    __shared__ float warp_sum[32];
    warp_sum[threadIdx.y] = 0.0f;
    atomicAdd(&warp_sum[threadIdx.y], sum);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = warp_sum[threadIdx.y];
        if (__bfloat162float(vec_sparse_val) > threshold){
            sum = sum * __bfloat162float(vec_sparse_val);
        }
        *res_p = __float2bfloat16(sum);
    }
}

void launch_ffn_fuse_23(nv_bfloat16 *vec_sparse, nv_bfloat16 *vec_input,
                        nv_bfloat16 *mat_up, nv_bfloat16 *res, unsigned int mat_row,
                        unsigned int mat_col, float threshold)
{
    dim3 grid_dim(1, mat_col / 32);
    dim3 block_dim(32, 32, 1);

    ffn_fuse_23<<<grid_dim, block_dim>>>(vec_sparse, vec_input, mat_up, res,
                                   mat_row, mat_col, threshold);

}