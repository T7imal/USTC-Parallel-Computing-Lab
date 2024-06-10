#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

int main()
{
    cudaDeviceProp deviceProp;
    int deviceCount;
    cudaError_t cudaError;
    cudaError = cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        cudaError = cudaGetDeviceProperties(&deviceProp, i);

        cout << "设备 " << i + 1 << " 的主要属性： " << endl;
        cout << "设备显卡型号： " << deviceProp.name << endl;
        cout << "设备全局内存总量（以MB为单位）： " << deviceProp.totalGlobalMem / 1024 / 1024 << endl;
        cout << "设备上一个线程块（Block）中可用的最大共享内存（以KB为单位）： " << deviceProp.sharedMemPerBlock / 1024
             << endl;
        cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << deviceProp.regsPerBlock << endl;
        cout << "设备上一个线程块（Block）可包含的最大线程数量： " << deviceProp.maxThreadsPerBlock << endl;
        cout << "设备的计算功能集（Compute Capability）的版本号： " << deviceProp.major << "." << deviceProp.minor
             << endl;
        cout << "设备上多处理器的数量： " << deviceProp.multiProcessorCount << endl;
    }
    getchar();
    return 0;
}