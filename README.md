# Efficient GPU Operators for ReLU-Activated LLMs

This is the source codes for our two sparse GPU operators mentioned in paper *ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models* ([link](https://arxiv.org/pdf/2402.13516.pdf)), tailored for the ReLU-activated FFNs in LLM.

### Background

The utilization of activation sparsity, namely the existence of considerable weakly-contributed elements among activation outputs, is a promising method for inference acceleration of large language models (LLMs). Concretely, acceleration methods based on activation sparsity usually achieve higher inference speed by making wiser resource allocation and computation policies to avoid resource waste on these weakly-contributed parameters. However, existing acceleration frameworks are mostly approximate algorithms, which risk potential inference inaccuracies caused by invalid predictions made by activation predictors (e.g., [Deja Vu](https://proceedings.mlr.press/v202/liu23am/liu23am.pdf) and [PowerInfer](https://arxiv.org/pdf/2312.12456.pdf)).

Therefore, to achieve acceleration without inference inaccuracies and test the practical speedup effects of ReLU-activated LLMs with higher sparsity, we implement two hardware-efficient sparse GPU operators with system-level optimizations, such as operator fusion, coalesced memory access, and vectorization, thereby exploiting input-side and output-side sparsity.

### Methodology

Given the hidden dimension $`d_{model}`$ and the FFN intermediate dimension $`d_{ff}`$, the computation process of a gated FFN can be formalized as:
```math
\mathbf{s} = \sigma(\mathbf{x} \mathbf{W}_s^T), \quad \mathbf{x}_1 = \mathbf{s} \odot (\mathbf{x} \mathbf{W}_1^T),\quad
    \text{FFN}(\mathbf{x}) = \mathbf{x}_1  \mathbf{W}_2^T,
```
where $`\mathbf{x}\in\mathbb{R}^{d_{model}}`$, $`\mathbf{s}, \mathbf{x}_1\in\mathbb{R}^{d_{ff}}`$, $`\sigma`$, and $`\odot`$ denote the input hidden states, the gating scores, the intermediate outputs, the activation function, and the element-wise multiplication respectively. $`\mathbf{W}_s,\mathbf{W}_1\in\mathbb{R}^{d_{ff} \times d_{model}}`$ and $`\mathbf{W}_2\in\mathbb{R}^{d_{model} \times d_{ff}}`$ are learnable weights.

We reorganize a ReLU-activated gated FFN into three major steps and our two operators, called **Operator Step (2)** `ffn_23` and **Operator Step (3)** `ffn_4`, are responsible for the step (2) and (3) respectively:

(1) A dense matrix-vector multiplication operator $`\mathbf{x} \mathbf{W}_s^T`$ which can be directly supported by vendor libraries such as cuBLAS;
(2) A fused operator of ReLU and $`\mathbf{s} \odot (\mathbf{x} \mathbf{W}_1^T)`$ with output-side sparsity;
(3) A sparse matrix-vector multiplication operator $`\mathbf{x}_1 \mathbf{W}_2^T`$ with input-side sparsity.

Codes for Operator Step (2) and Operator Step (3) are included in `kernel/formula_23_kernel.cu` and `kernel/formula_4_kernel.cu` respectively. For more implementation details, refer to Appendix C of [paper](https://arxiv.org/pdf/2402.13516.pdf).

### Results

To test the practical acceleration effects of ReLU-activated LLMs with the above operators applied, we measure the average single-step wall-clock time spent by our two sparse GPU operators, which are responsible for step (2) and step (3) respectively. Major results are shown as follows, refer to Section 4.3 of [paper](https://arxiv.org/pdf/2402.13516.pdf) for more details. The ProSparse LLaMA2 models, which have ReLU-based high activation sparsity and comparable performance to original Swish-activated LLaMA2 versions, are available at the following links: [7B](https://huggingface.co/SparseLLM/prosparse-llama-2-7b) and [13B](https://huggingface.co/SparseLLM/prosparse-llama-2-13b).

|          Setting          | Average<br>Sparsity | Step (2)<br>Time | Step (2)<br>Speedup | Step (3)<br>Time | Step (3)<br>Speedup |
|:-------------------------:|:-------------------:|:----------------:|:-------------------:|:----------------:|:----------------:|
|       ReluLLaMA-7B        |        66.98        |      67.12       |        1.35         |      63.00       |       1.32       |
|      Vanilla ReLU-7B      |        66.04        |      67.85       |        1.33         |      63.28       |       1.31       |
|      Fixed $`L_1`$-7B       |        91.46        |      40.99       |        2.21         |      54.19       |       1.53       |
|   **ProSparse-7B**$`^*`$    |        88.11        |      46.66       |        1.94         |      55.56       |       1.49       |
|     **ProSparse-7B**      |        89.32        |      45.38       |        2.00         |      55.05       |       1.51       |
|       ReluLLaMA-13B       |        71.56        |      69.92       |        1.88         |      75.47       |       1.51       |
|   **ProSparse-13B**$`^*`$   |        87.97        |      55.29       |        2.38         |      67.50       |       1.68       |
|     **ProSparse-13B**     |        88.80        |      53.78       |        2.44         |      66.73       |       1.70       |

`Time` means the average wall-clock time (us) cost by each step with our sparse GPU operators, and `Speedup` is the speedup ratio to the setting without operators. The average time for step (2) and (3) without sparse GPU operators is about **90.55 and 82.92 (us) for 7B, 131.36 and 113.68 (us) for 13B** respectively under all sparsity.

As demonstrated by the above results, higher activation sparsity can make accurate algorithms based on GPU operators more efficient. Besides, our two sparse GPU operators also display satisfactory speedup ratios up to 2.44 and 1.70 respectively with better acceleration effects for larger models.

### Install

Use the following command to install `ffn_23` for Operator Step (2) and `ffn_4` for Operator Step (3).

```bash
python setup.py install
```

**Note**: In some environments, the above command may not work. Under such cases, enter the root folder and then run `pip install .` twice after annotating the two `setup` function calls in `setup.py` one after the other to install the two operators one by one.

### Usage

See `run_test_23.py` and `run_test_4.py`.

### Attention: FATReLU support

Note that our Operator Step (2) supports FATReLU, a non-zero threshold ReLU variant:
```math
\sigma(x)=
    \begin{cases}
    x \quad \mathrm{when}\ x \geq T, \\
    0 \quad \mathrm{otherwise},
    \end{cases}
```
where $`T>0`$ is a positive threshold. Remember to specify $`T`$ as the last parameter of a call to `ffn_23`, use 0 for vanilla ReLU.

### Attention: Data Types

The default data type used in these codes is **bfloat16**. Nevertheless, other data types can be easily supported through an overall substitution of data types in source codes.

### Attention: Dimensions

We found significant performance improvement if the dimensions are pre-defined as fixed values in Operator Step (3). The default dimensions are fixed to the settings of LLaMA-7B. If other dimensions (e.g., LLaMA-13B) have to be supported, just make the following modification in `kernel/formula_4_kernel.cu`.

```c++
// Default setting for LLaMA-7B
__global__ void ffn_4(nv_bfloat16 *mat, nv_bfloat16 *vec, nv_bfloat16 *res,
                      unsigned int mat_row, unsigned int mat_col)
{
    mat_row = 11008;
    mat_col = 4096;

    float sum = 0;
    ......
}

// Example: change to the setting of LLaMA-13B
__global__ void ffn_4(nv_bfloat16 *mat, nv_bfloat16 *vec, nv_bfloat16 *res,
                      unsigned int mat_row, unsigned int mat_col)
{
    mat_row = 13824;
    mat_col = 5120;

    float sum = 0;
    ......
}
```

### Citation

Please kindly cite using the following BibTeX:

```bibtex
@article{song2024prosparse,
  title={{ProSparse}: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models},
  author={Song, Chenyang and Han, Xu and Zhang, Zhengyan and Hu, Shengding and Shi, Xiyu and Li, Kuai and Chen, Chen and Liu, Zhiyuan and Li, Guangli and Yang, Tao and Sun, Maosong},
  year={2024},
  journal={arXiv preprint arXiv:2402.13516},
  url={https://arxiv.org/pdf/2402.13516.pdf}
}
```
