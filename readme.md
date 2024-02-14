[ref] https://github.dev/godweiyang/NN-CUDA-Example/tree/master


```bash
# 进入torch环境，例如：
conda deactivate
conda activate pt2

# 生成动态链接库，同时将ffn_4/ffn_23添加为python的模块，直接import xxx来调用
python setup.py install 

python run_test.py # 公式4
python run_test-for-all.py # 公式4（对100行真实sparse vec的性能测试）
python run_test_23.py # 公式23
python run_test-for-all-23.py # 公式23（对100行真实sparse vec的性能测试）
```
备用：
`CUDA_VISIBLE_DEVICES=7 nsys profile --trace=cuda --stats=true python run_test.py`; 
`export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__" `