#include <torch/extension.h>
#include "formula_23.h"

void torch_launch_ffn_fuse_23(torch::Tensor &vec_sparse,
                              torch::Tensor &vec_input,
                              torch::Tensor &mat_up,
                              torch::Tensor &res, unsigned int batch_size,
                              unsigned int mat_row, unsigned int mat_col, float threshold = 0.)
{
    launch_ffn_fuse_23((nv_bfloat16 *)vec_sparse.data_ptr(),
                       (nv_bfloat16 *)vec_input.data_ptr(),
                       (nv_bfloat16 *)mat_up.data_ptr(),
                       (nv_bfloat16 *)res.data_ptr(), batch_size, mat_row, mat_col, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("torch_launch_ffn_fuse_23",
          &torch_launch_ffn_fuse_23,
          "ffn_fuse_23 kernel warpper");
}
