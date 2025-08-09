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

