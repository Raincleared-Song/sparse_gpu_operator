import time
import numpy as np
import torch
import ffn_23

nwarmup = 10
ntest = 100

def show_time(func):
    times = list()
    res = None
    # GPU warm up
    for _ in range(nwarmup):
        res = func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        res = func()
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
        if (abs(b) == 0 and abs(a) > tolerance or abs(b) > 0 and abs(a - b) / abs(b) > tolerance):
            print(f"Index {index}: diff = {a-b}")
            return False
    return True

mat_row = 4096
mat_col = 11008
threshold = 0.

file_path = 'pytorch/sparse_vec.npy'
data = np.load(file_path)


# first_row = data[18, :]
# vec_sparse = torch.tensor(first_row, device="cuda:0")

# assert vec_sparse.shape == (mat_col,), f"Expected shape (mat_col,), but got {first_row.shape}"
vec_sparse = torch.zeros(mat_col, device="cuda:0", dtype=torch.float16)
vec = torch.rand(mat_row, device="cuda:0", dtype=torch.bfloat16)
# mat = torch.zeros(mat_row, mat_col, device="cuda:0", dtype=torch.bfloat16)
cuda_res = torch.zeros(mat_col, device="cuda:0", dtype=torch.bfloat16)


def run_torch():
    res = torch.matmul(vec, mat)
    res = res * vec_sparse
    # res[vec_sparse == 0] = 0
    return res

def run_cuda():
    ffn_23.torch_launch_ffn_fuse_23(vec_sparse, vec, mat, cuda_res, mat_row, mat_col, threshold)
    return cuda_res


print(f"index,num_nonzero_elements,%,cuda_time,torch_time")
for i in range(100):
    row = data[i, :]
    vec_sparse = torch.tensor(row, device="cuda:0")
    vec_sparse = vec_sparse.to(dtype=torch.bfloat16)
    assert vec_sparse.shape == (mat_col,), f"Expected shape (mat_col,), but got {row.shape}"

    nonzero_indices = torch.nonzero(vec_sparse)
    num_nonzero_elements = nonzero_indices.size(0)
    mat = torch.rand(mat_row, mat_col, device="cuda:0", dtype=torch.bfloat16)
    torch_time, torch_res = show_time(run_torch)
    mat = mat.t().contiguous()
    cuda_time, cuda_res = show_time(run_cuda)

    print(f"{i},{num_nonzero_elements},{round(num_nonzero_elements/mat_col*100, 3)},{np.mean(cuda_time)},{np.mean(torch_time)}")

    tolerance = 0.01
    if not compare_tensors(cuda_res, torch_res, tolerance):
        from IPython import embed
        embed()
        exit()
