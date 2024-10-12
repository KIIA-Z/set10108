#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace std;

// 5x5 gaussian kernel radius
constexpr int RADIUS = 2;

__global__ void smooth(float* values_in, float* values_out, int width, int height) {
    constexpr float KERNEL[] = {
        0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
        0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
        0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
        0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
        0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
    };

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    float tmp = 0.0f;
    for (int oy = -RADIUS; oy <= RADIUS; ++oy) {
        int ynb = min(max(y + oy, 0), height - 1);
        int kernel_row = oy + RADIUS;
        for (int ox = -RADIUS; ox <= RADIUS; ++ox) {
            int xnb = min(max(x + ox, 0), width - 1);
            int kernel_col = ox + RADIUS;
            float value = values_in[xnb + ynb * width] * KERNEL[kernel_col + kernel_row * 5];
            tmp += value;
        }
    }
    values_out[x + y * width] = tmp;
}

std::vector<float> load_image_to_grayscale(const char* filename, int& width, int& height) {
    int n;
    auto imgdata = (uint8_t*)stbi_load(filename, &width, &height, &n, 0);
    vector<float> values(width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int o = x + y * width;
            if (n >= 3) {
                float r = imgdata[o * n] / 255.0f;
                float g = imgdata[o * n + 1] / 255.0f;
                float b = imgdata[o * n + 2] / 255.0f;
                values[o] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            }
            else if (n == 1) {
                values[o] = imgdata[o] / 255.0f;
            }
        }
    stbi_image_free(imgdata);
    return values;
}

void save_grayscale_png(const char* filename, const std::vector<float>& values, int width, int height) {
    std::vector<uint8_t> imgdata(width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            imgdata[x + y * width] = uint8_t(values[x + y * width] * 255);
    stbi_write_png(filename, width, height, 1, imgdata.data(), width);
}

__constant__ float d_kernel[225]; // Large enough for radius up to 7

__global__ void gaussianSmoothingKernel2D(const float* input, float* output, int width, int height, int radius) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int kernelSize = 2 * radius + 1;

    for (int oy = -radius; oy <= radius; ++oy) {
        int ynb = min(max(y + oy, 0), height - 1);
        int kernel_row = oy + radius;
        for (int ox = -radius; ox <= radius; ++ox) {
            int xnb = min(max(x + ox, 0), width - 1);
            int kernel_col = ox + radius;
            sum += input[xnb + ynb * width] * d_kernel[kernel_col + kernel_row * kernelSize];
        }
    }
    output[x + y * width] = sum;
}

void generateGaussianKernel(float* h_kernel, int radius, float sigma) {
    int kernelSize = 2 * radius + 1;
    float sum = 0.0f;

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            int index = (y + radius) * kernelSize + (x + radius);
            h_kernel[index] = expf(-(x * x + y * y) / (2 * sigma * sigma));
            sum += h_kernel[index];
        }
    }

    for (int i = 0; i < kernelSize * kernelSize; ++i) {
        h_kernel[i] /= sum;
    }
}

int main(int argc, char** argv) {

    std::string filename_in = "C:/Users/kia/source/repos/set10108/labs/unit3/97735.png"; // Change this to your actual image path
    if (argc > 1) {
        filename_in = argv[1];
    }

    //this is for entering a image path throught terminal
   /* if (argc == 1) {
        printf("Please provide an image filename as the argument (full path). The program will create a new image with the suffix _out.png and will place it alongside the original image");
        exit(0);
    }
    */

    int width, height;
    //std::string filename_in = argv[1];
    auto h_values_in = load_image_to_grayscale(filename_in.c_str(), width, height);
    const size_t NUM_PIXELS = h_values_in.size();
    const size_t NUM_BYTES = sizeof(float) * NUM_PIXELS;

    vector<float> h_values_out(NUM_PIXELS);

    int radius = 3; // You can change the radius here
    float sigma = 1.0f;
    float h_kernel[225]; // Adjust if you anticipate larger kernels

    generateGaussianKernel(h_kernel, radius, sigma);

    cudaMemcpyToSymbol(d_kernel, h_kernel, (2 * radius + 1) * (2 * radius + 1) * sizeof(float));

    float* d_values_in = nullptr;
    float* d_values_out = nullptr;

    cudaMalloc((void**)&d_values_in, NUM_BYTES);
    cudaMalloc((void**)&d_values_out, NUM_BYTES);

    cudaMemcpy(d_values_in, h_values_in.data(), NUM_BYTES, cudaMemcpyHostToDevice);

    dim3 blockDim = { 32, 32, 1 };
    dim3 gridDim = { (unsigned(width) + 31) / 32, (unsigned(height) + 31) / 32, 1 };
    gaussianSmoothingKernel2D << <gridDim, blockDim >> > (d_values_in, d_values_out, width, height, radius);

    cudaMemcpy(h_values_out.data(), d_values_out, NUM_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(d_values_in);
    cudaFree(d_values_out);

    auto filename_out = filename_in.substr(0, filename_in.size() - 4) + "_out.png";
    save_grayscale_png(filename_out.c_str(), h_values_out, width, height);

    std::cout << "Smoothing complete! The output image is saved as " << filename_out << std::endl;

    // Print some values for visualization
    std::cout << "Sample of smoothed values (printed in a 10x10 grid):" << std::endl;
    for (int i = 0; i < min(10, height); ++i) {
        for (int j = 0; j < min(10, width); ++j) {
            std::cout << h_values_out[j + i * width] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}