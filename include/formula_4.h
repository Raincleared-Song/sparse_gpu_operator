#include <cuda_bf16.h>
void launch_ffn_4(nv_bfloat16 *mat, nv_bfloat16 *vec, nv_bfloat16 *res,
                  unsigned int mat_row, unsigned int mat_col);