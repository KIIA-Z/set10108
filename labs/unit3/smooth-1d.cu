#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void gaussianSmoothingKernel(const float* input, float* output, const float* kernel, int dataSize, int radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        float sum = 0.0f;
        int kernelSize = 2 * radius + 1;

        for (int i = -radius; i <= radius; ++i) {
            int neighborIdx = idx + i;
            if (neighborIdx >= 0 && neighborIdx < dataSize) {
                sum += input[neighborIdx] * kernel[radius + i];
            }
        }

        output[idx] = sum;
    }
}

void generateGaussianKernel(float* kernel, int radius, float sigma) {
    int kernelSize = 2 * radius + 1;
    float sum = 0.0f;

    for (int i = -radius; i <= radius; ++i) {
        kernel[radius + i] = expf(-(i * i) / (2 * sigma * sigma));
        sum += kernel[radius + i];
    }

    for (int i = 0; i < kernelSize; ++i) {
        kernel[i] /= sum;
    }
}

int main() {
    const int dataSize = 512;
    const int radius = 5; // Arbitrary radius value
    const float sigma = 1.0f;

    std::vector<float> input(dataSize);
    std::vector<float> output(dataSize);
    int kernelSize = 2 * radius + 1;
    std::vector<float> kernel(kernelSize);

    // Initialize input with random values
    for (int i = 0; i < dataSize; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    generateGaussianKernel(kernel.data(), radius, sigma);

    float* d_input, * d_output, * d_kernel;
    size_t dataSizeBytes = dataSize * sizeof(float);
    size_t kernelSizeBytes = kernelSize * sizeof(float);

    cudaMalloc(&d_input, dataSizeBytes);
    cudaMalloc(&d_output, dataSizeBytes);
    cudaMalloc(&d_kernel, kernelSizeBytes);

    cudaMemcpy(d_input, input.data(), dataSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), kernelSizeBytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (dataSize + blockSize - 1) / blockSize;

    gaussianSmoothingKernel << <numBlocks, blockSize >> > (d_input, d_output, d_kernel, dataSize, radius);

    cudaMemcpy(output.data(), d_output, dataSizeBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    std::cout << "Gaussian smoothed values:\n";
    for (int i = 0; i < dataSize; ++i) {
        std::cout << output[i] << " ";
        if ((i + 1) % 10 == 0) { // Print 10 values per line
            std::cout << std::endl;
        }
    }

    return 0;
}
