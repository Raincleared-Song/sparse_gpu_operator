import time
import numpy as np
import torch
import ffn_4

ntest = 1000

def show_time(func):
    times = list()
    res = None
    # GPU warm up
    for _ in range(10):
        res = func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time)*1e6)
    return times, res

def compare_tensors(res_cuda, res_torch, tolerance):
    if res_cuda.shape != res_torch.shape:
        print("Tensor shapes are different.")
        return False
    res_cuda_list = res_cuda.tolist()
    res_torch_list = res_torch.tolist()

    for index, (a, b) in enumerate(zip(res_cuda_list, res_torch_list)):
        if (abs(a - b) > tolerance):
            print(f"Index {index}: diff = {a-b}")

mat_row = 11008;
mat_col = 4096;

# mat_row = 256;
# mat_col = 512;
# mat = torch.rand(mat_row, mat_col, device="cuda:0", dtype=torch.float32)
# vec = torch.rand(mat_row, device="cuda:0", dtype=torch.float32)
# ffn_4.torch_launch_ffn_4(mat.to(dtype=torch.bfloat16), vec.to(dtype=torch.bfloat16), res_cuda, mat_row, mat_col)

file_path = 'pytorch/sparse_vec.npy'
data = np.load(file_path)

mat = torch.rand(mat_row, mat_col, device="cuda:0", dtype=torch.bfloat16)
vec = torch.zeros(mat_row, device="cuda:0", dtype=torch.bfloat16)
cuda_res = torch.zeros(mat_col, device="cuda:0", dtype=torch.bfloat16)


def run_cuda():
    ffn_4.torch_launch_ffn_4(mat, vec, cuda_res, mat_row, mat_col)
    return cuda_res

def run_torch():
    res = torch.matmul(vec, mat)
    return res


for i in range(100):
    row = data[i, :]
    vec = torch.tensor(row, device="cuda:0")
    vec = vec.to(dtype=torch.bfloat16)
    assert vec.shape == (mat_row,), f"Expected shape (mat_row,), but got {row.shape}"

    nonzero_indices = torch.nonzero(vec)
    num_nonzero_elements = nonzero_indices.size(0)
    cuda_time, cuda_res = show_time(run_cuda)
    torch_time, torch_res = show_time(run_torch)

    print(f"{i},{num_nonzero_elements},{round(num_nonzero_elements/11008*100, 3)},{np.mean(cuda_time)},{np.mean(torch_time)}")

    # tolerance = 1;
    # compare_tensors(cuda_res, torch_res, tolerance)
