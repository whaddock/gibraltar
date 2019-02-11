# K40 GPU Facts
[whaddock@iprogress-phi deviceQuery]$ ./deviceQuery 
./deviceQuery Starting...

## CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Tesla K40m"
* CUDA Driver Version / Runtime Version          9.2 / 9.2
* CUDA Capability Major/Minor version number:    3.5
* Total amount of global memory:                 11441 MBytes (11996954624 bytes)
* (15) Multiprocessors, (192) CUDA Cores/MP:     2880 CUDA Cores
* GPU Max Clock rate:                            745 MHz (0.75 GHz)
* Memory Clock rate:                             3004 Mhz
* Memory Bus Width:                              384-bit
* L2 Cache Size:                                 1572864 bytes
* Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
* Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
* Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
* Total amount of constant memory:               65536 bytes
* Total amount of shared memory per block:       49152 bytes
* Total number of registers available per block: 65536
* Warp size:                                     32
* Maximum number of threads per multiprocessor:  2048
* Maximum number of threads per block:           1024
* Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
* Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
* Maximum memory pitch:                          2147483647 bytes
* Texture alignment:                             512 bytes
* Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
* Run time limit on kernels:                     No
* Integrated GPU sharing Host Memory:            No
* Support host page-locked memory mapping:       Yes
* Alignment requirement for Surfaces:            Yes
* Device has ECC support:                        Enabled
* Device supports Unified Addressing (UVA):      Yes
* Device supports Compute Preemption:            No
* Supports Cooperative Kernel Launch:            No
* Supports MultiDevice Co-op Kernel Launch:      No
* Device PCI Domain ID / Bus ID / location ID:   0 / 130 / 0
* Compute Mode:
  * Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) 

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 9.2, CUDA Runtime Version = 9.2, NumDevs = 1
* Result = PASS
