> The information presented in here is not complete/final. I am actively researching 
things and I will regularly update things as my understanding improves and abstraction
layers are removed.

Nvidia RTX 2060 (Turing microarchitecture):
- 30 SMs
- 1920 cuda cores
- 64 cuda cores per SM

Each cuda core executes 1 thread's instruction per cycle (in optimal conditions). 
Threads are run in `warps` of 32. So the following can be said:
- 1 SM has 64 cuda cores = 64/32 = 2 warps per cycle
- 30 SMs = 30x2 = 60 warps per cycle
- Total threads GPU can handle = 60x32 = 1920 threads per cycle = 1920 cuda cores

However the `RTX 2060` has a max number of resident threads per SM of `1024`, 
these are in what I would describe as a waiting room so when one thread stalls or 
waits for data to come back, another thread can immediatly start running.

In the code example the kernel is launched with:
- 262144 threads
- 32 threads per block
- we get 8192 warps in total
To make all of this run in 1 cycle we would need 262144 cuda cores or `4096 SMs`.
In reality on a `RTX 2060` the maximum number of warps we can run in 1 cycle is 60,
so for 8192 warps we would need `137 cycles minimum` (8192/60). 

[look up]
latency hiding?  

`main.cu` is a small program to test how the number of threads used per block can 
affect the speed of the kernel computation.  
The results show that as you increase the thread number per block the speed doubles 
until it reaches `32` threads per block, which is the current warp size. 
After that the speed stays the same.

```shell
Total number of threads: 262144

Threads per block       Time (ms)
1                       2317.11
2                       1124.40
4                       562.87
8                       282.76
16                      142.27
32                      72.60
64                      72.66
128                     72.63
256                     73.03
512                     75.08
1024                    75.07
```

Another thing I found was that instead of doing a cold start for the kernel on the GPU,
having a warm up run provides performance gains on that first run the kernel makes.

Code for the warm up run:
```cuda
for (int i = 0; i < 100; i++) {
    kernel<<<1, 1>>>(data_d);
}
```

Without warm up run:
```shell
Total number of threads: 262144

Threads per block       Time (ms)
16                      192.44
```

With warm up:
```shell
Total number of threads: 262144

Threads per block       Time (ms)
16                      140.60
```

If the number of threads per block is higher than `1024`, we get an error saying:
```shell
Total number of threads: 262144

Threads per block       Time (ms)
Error launching the kernel: invalid configuration argument
1025                    0.00
```
This is because the `RTX 2060` has a compute capability of 7.5, which specifies the 
max number of threads per block as 1024. This has been the case since March 2010, once
compute capability reached 2.x and higher.

There is no clear 1-to-1 mapping for the blocks in the grid and the SMs in the GPU.
It is a `dynamic assignment` where the CUDA global scheduler distributes blocks to the 
SMs in a circular(round-robin) fashion. SMs will take blocks in turns until all the 
blocks in the grid are assigned to a SM. 
This approach balances the workload, ensuring most SMs receive some work even if light
instead of leaving SMs idle while others are overloaded.

## References
1. [Nvidia TU106 breakdown](https://www.techpowerup.com/gpu-specs/nvidia-tu106.g875) 
2. [Turing wikipedia](https://en.wikipedia.org/wiki/Turing_(microarchitecture)) 
3. [GeForce RTX 20 series](https://en.wikipedia.org/wiki/GeForce_RTX_20_series) 
4. [Thread Block (CUDA programming)](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)) 
