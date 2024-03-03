// #include <cuda_fp16.h>
#include <cuda_bf16.h>
void launch_ffn_fuse_23(nv_bfloat16 *vec_sparse, nv_bfloat16 *vec_input,
                        nv_bfloat16 *mat_up, nv_bfloat16 *res, unsigned int batch_size,
                        unsigned int mat_row, unsigned int mat_col, float threshold = 0);