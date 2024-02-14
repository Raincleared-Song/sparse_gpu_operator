#include <torch/extension.h>
#include "formula_4.h"

void torch_launch_ffn_4(torch::Tensor &mat,
                        torch::Tensor &vec,
                        torch::Tensor &res,
                        int mat_row, int mat_col)
{
    launch_ffn_4((nv_bfloat16 *)mat.data_ptr(),
                 (nv_bfloat16 *)vec.data_ptr(),
                 (nv_bfloat16 *)res.data_ptr(), mat_row, mat_col);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("torch_launch_ffn_4",
          &torch_launch_ffn_4,
          "ffn_4 kernel warpper");
}
