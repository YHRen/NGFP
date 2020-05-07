# Performance analysis of NFP generation code

Using pretrained model to generate NFP. On my 8700k with 1080 GPU, it can reach 750 its/s, but on a server-grade xeon CPU, it is only 600 its/s.


### On 8700k with GTX 1080
```
--------------------------------------------------------------------------------
  Environment Summary
--------------------------------------------------------------------------------
PyTorch 1.4.0 compiled w/ CUDA 10.1
Running with Python 3.8 and 

`pip3 list` truncated output:
torchfile==0.1.0
--------------------------------------------------------------------------------
  cProfile output
--------------------------------------------------------------------------------
         172246818 function calls (172209983 primitive calls) in 156.669 seconds

   Ordered by: internal time
   List reduced from 634 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      399   59.245    0.148  116.125    0.291 /home/yren/Documents/Projects/BNL/COVID19/NGFP/NeuralGraph/preprocessing.py:32(tensorise_smiles)
    99782   25.499    0.000   25.499    0.000 generate_nfp.py:48(is_valid_smile_for_NFP)
  2799044   13.983    0.000   33.314    0.000 /home/yren/Documents/Projects/BNL/COVID19/NGFP/NeuralGraph/feature.py:12(atom_features)
 13995220   13.498    0.000   19.331    0.000 /home/yren/Documents/Projects/BNL/COVID19/NGFP/NeuralGraph/feature.py:5(one_of_k_encoding)
    99782    9.472    0.000    9.472    0.000 /home/yren/Documents/Projects/BNL/COVID19/NGFP/NeuralGraph/preprocessing.py:94(<listcomp>)
100765584    5.833    0.000    5.833    0.000 /home/yren/Documents/Projects/BNL/COVID19/NGFP/NeuralGraph/feature.py:9(<lambda>)
  3021442    5.616    0.000    9.162    0.000 /home/yren/Documents/Projects/BNL/COVID19/NGFP/NeuralGraph/feature.py:29(bond_features)
  6075410    5.335    0.000    5.335    0.000 {built-in method numpy.array}
 12774513    4.911    0.000    4.911    0.000 {method 'format' of 'str' objects}
 11610000    2.385    0.000    6.811    0.000 generate_nfp.py:146(<genexpr>)
     2825    1.952    0.001    1.952    0.001 {method 'to' of 'torch._C._TensorBase' objects}
   100191    1.019    0.000    8.559    0.000 {method 'join' of 'str' objects}
       58    0.996    0.017    0.996    0.017 {built-in method __new__ of type object at 0x55c60c7f5ac0}
        1    0.953    0.953  156.669  156.669 generate_nfp.py:1(<module>)
  3021452    0.747    0.000    0.747    0.000 {built-in method builtins.max}


--------------------------------------------------------------------------------
  autograd profiler output (CUDA mode)
--------------------------------------------------------------------------------
        top 15 events sorted by cpu_time_total

	Because the autograd profiler uses the CUDA event API,
	the CUDA time column reports approximately max(cuda_time, cpu_time).
	Please ignore this output if your code does not use CUDA.

--------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
Name            Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     CUDA total %     CUDA total       CUDA time avg    Number of Calls  Input Shapes                         
--------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
to              7.61%            3.029ms          7.61%            3.029ms          3.029ms          5.46%            544.000us        544.000us        1                []                                   
matmul          7.60%            3.024ms          7.60%            3.024ms          3.024ms          19.20%           1.912ms          1.912ms          1                []                                   
contiguous      7.49%            2.980ms          7.49%            2.980ms          2.980ms          17.51%           1.744ms          1.744ms          1                []                                   
to              7.43%            2.958ms          7.43%            2.958ms          2.958ms          5.16%            514.000us        514.000us        1                []                                   
to              6.97%            2.774ms          6.97%            2.774ms          2.774ms          4.83%            481.281us        481.281us        1                []                                   
to              6.71%            2.672ms          6.71%            2.672ms          2.672ms          4.79%            476.812us        476.812us        1                []                                   
to              6.60%            2.625ms          6.60%            2.625ms          2.625ms          4.75%            473.000us        473.000us        1                []                                   
to              6.41%            2.553ms          6.41%            2.553ms          2.553ms          4.67%            465.000us        465.000us        1                []                                   
to              6.27%            2.495ms          6.27%            2.495ms          2.495ms          5.06%            504.000us        504.000us        1                []                                   
to              6.25%            2.488ms          6.25%            2.488ms          2.488ms          4.66%            464.000us        464.000us        1                []                                   
to              6.21%            2.470ms          6.21%            2.470ms          2.470ms          4.74%            472.000us        472.000us        1                []                                   
to              6.17%            2.456ms          6.17%            2.456ms          2.456ms          4.50%            448.000us        448.000us        1                []                                   
to              6.10%            2.429ms          6.10%            2.429ms          2.429ms          4.86%            484.000us        484.000us        1                []                                   
to              6.09%            2.425ms          6.09%            2.425ms          2.425ms          5.00%            498.000us        498.000us        1                []                                   
to              6.09%            2.423ms          6.09%            2.423ms          2.423ms          4.80%            478.000us        478.000us        1                []                                   
--------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
Self CPU time total: 39.803ms
CUDA time total: 9.958ms
```


### On Xeon 8168 with V100

```
--------------------------------------------------------------------------------
  Environment Summary
--------------------------------------------------------------------------------
PyTorch 1.3.1 compiled w/ CUDA 10.0.130
Running with Python 3.7 and 

`pip list` truncated output:
numpy==1.17.4
torch==1.3.1
torchvision==0.4.2
--------------------------------------------------------------------------------
  cProfile output
--------------------------------------------------------------------------------
         172224391 function calls (172187526 primitive calls) in 190.198 seconds

   Ordered by: internal time
   List reduced from 623 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      399   69.786    0.175  136.772    0.343 /home/yren/covid19/NGFP/NeuralGraph/preprocessing.py:32(tensorise_smiles)
    99782   34.027    0.000   34.027    0.000 generate_nfp.py:48(is_valid_smile_for_NFP)
  2799044   16.491    0.000   38.797    0.000 /home/yren/covid19/NGFP/NeuralGraph/feature.py:12(atom_features)
 13995220   15.715    0.000   22.306    0.000 /home/yren/covid19/NGFP/NeuralGraph/feature.py:5(one_of_k_encoding)
    99782   10.641    0.000   10.641    0.000 /home/yren/covid19/NGFP/NeuralGraph/preprocessing.py:94(<listcomp>)
  3021442    7.075    0.000   11.437    0.000 /home/yren/covid19/NGFP/NeuralGraph/feature.py:29(bond_features)
100765584    6.591    0.000    6.591    0.000 /home/yren/covid19/NGFP/NeuralGraph/feature.py:9(<lambda>)
  6075410    6.583    0.000    6.583    0.000 {built-in method numpy.array}
 12774933    6.166    0.000    6.166    0.000 {method 'format' of 'str' objects}
       27    3.418    0.127    3.418    0.127 {built-in method __new__ of type object at 0x55c88d36c240}
 11610000    2.622    0.000    8.159    0.000 generate_nfp.py:146(<genexpr>)
     2825    1.605    0.001    1.605    0.001 {method 'to' of 'torch._C._TensorBase' objects}
        1    1.106    1.106  190.198  190.198 generate_nfp.py:1(<module>)
   100201    1.077    0.000   10.143    0.000 {method 'join' of 'str' objects}
     6585    0.894    0.000    0.934    0.000 /home/yren/anaconda3/envs/torch1.3/lib/python3.7/site-packages/numpy/lib/arraypad.py:88(_pad_simple)


--------------------------------------------------------------------------------
  autograd profiler output (CUDA mode)
--------------------------------------------------------------------------------
        top 15 events sorted by cpu_time_total

	Because the autograd profiler uses the CUDA event API,
	the CUDA time column reports approximately max(cuda_time, cpu_time).
	Please ignore this output if your code does not use CUDA.

-----------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
Name               Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     CUDA total %     CUDA total       CUDA time avg    Number of Calls  Input Shapes                         
-----------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
to                 38.04%           81.588ms         38.04%           81.588ms         81.588ms         38.02%           81.496ms         81.496ms         1                []                                   
empty              38.02%           81.553ms         38.02%           81.553ms         81.553ms         38.00%           81.460ms         81.460ms         1                []                                   
to                 3.31%            7.096ms          3.31%            7.096ms          7.096ms          3.31%            7.096ms          7.096ms          1                []                                   
scalar_tensor      2.03%            4.349ms          2.03%            4.349ms          4.349ms          2.03%            4.352ms          4.352ms          1                []                                   
fill_              2.02%            4.326ms          2.02%            4.326ms          4.326ms          2.02%            4.332ms          4.332ms          1                []                                   
matmul             1.86%            3.990ms          1.86%            3.990ms          3.990ms          1.87%            4.008ms          4.008ms          1                []                                   
contiguous         1.82%            3.905ms          1.82%            3.905ms          3.905ms          1.82%            3.910ms          3.910ms          1                []                                   
to                 1.76%            3.778ms          1.76%            3.778ms          3.778ms          1.76%            3.778ms          3.778ms          1                []                                   
matmul             1.73%            3.708ms          1.73%            3.708ms          3.708ms          1.74%            3.736ms          3.736ms          1                []                                   
view               1.69%            3.630ms          1.69%            3.630ms          3.630ms          1.69%            3.632ms          3.632ms          1                []                                   
matmul             1.59%            3.419ms          1.59%            3.419ms          3.419ms          1.59%            3.408ms          3.408ms          1                []                                   
mm                 1.57%            3.367ms          1.57%            3.367ms          3.367ms          1.57%            3.376ms          3.376ms          1                []                                   
matmul             1.55%            3.331ms          1.55%            3.331ms          3.331ms          1.56%            3.344ms          3.344ms          1                []                                   
contiguous         1.51%            3.245ms          1.51%            3.245ms          3.245ms          1.52%            3.248ms          3.248ms          1                []                                   
matmul             1.49%            3.198ms          1.49%            3.198ms          3.198ms          1.49%            3.200ms          3.200ms          1                []                                   
-----------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
Self CPU time total: 214.481ms
CUDA time total: 214.376ms
```
