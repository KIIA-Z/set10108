#define __CL_ENABLE_EXCEPTIONS

#include <filesystem>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <CL/cl.h>
#include <CL/cl.hpp>

using namespace std;
using namespace cl;
using namespace std::chrono;


// Function to read the file into a vector of char
std::vector<char> read_file(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error: Could not read the file content." << std::endl;
        return {};
    }
    file.close();
    std::cout << "Successfully read " << buffer.size() << " bytes from the file." << std::endl;
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });
    return buffer;
}

// OpenCL kernel source
const char* kernelSource = R"(
__kernel void calcTokenOccurrences(
    __global const char* data,
    __global int* results,
    __global const char* tokens,
    __global const int* tokenOffsets,
    __global const int* tokenLengths,
    const int dataSize,
    const int numTokens
) {
    int id = get_global_id(0);
    if (id >= dataSize) return;

    for (int t = 0; t < numTokens; ++t) {
        int tokenLen = tokenLengths[t];
        int tokenIdx = tokenOffsets[t];
        int match = 1;
        for (int i = 0; i < tokenLen; ++i) {
            if (id + i >= dataSize || data[id + i] != tokens[tokenIdx + i]) {
                match = 0;
                break;
            }
        }
        if (match == 0) continue;

        int iPrefix = id - 1;
        if (iPrefix >= 0) {
            if (data[iPrefix] >= 'a' && data[iPrefix] <= 'z') continue;
        }

        int iSuffix = id + tokenLen;
        if (iSuffix < dataSize) {
            if (data[iSuffix] >= 'a' && data[iSuffix] <= 'z') continue;
        }

        atomic_inc(&results[t]);
    }
})";


int calc_token_occurrences(const std::vector<char>& data, const char* token)
{
    int numOccurrences = 0;
    int tokenLen = int(strlen(token));
    for (int i = 0; i< int(data.size()); ++i)
    {
        // test 1: does this match the token?
        auto diff = strncmp(&data[i], token, tokenLen);
        if (diff != 0)
            continue;

        // test 2: is the prefix a non-letter character?
        auto iPrefix = i - 1;
        if (iPrefix >= 0 && data[iPrefix] >= 'a' && data[iPrefix] <= 'z')
            continue;

        // test 3: is the prefix a non-letter character?
        auto iSuffix = i + tokenLen;
        if (iSuffix < int(data.size()) && data[iSuffix] >= 'a' && data[iSuffix] <= 'z')
            continue;
        ++numOccurrences;
    }
    return numOccurrences;
}

int main() {
    

    uint64_t total_ns_GPU = 0;
    uint64_t total_ns_CPU = 0;

    ofstream CPUdata_unop("CPUdata.csv", ofstream::out);
    ofstream GPUdata_unoptimized("GPUdata.csv", ofstream::out);


    std::vector<char> data = read_file("dataset/pride_and_prejudice.txt");
    int dataSize = data.size();

    //-------------------CPU/Original code Area---------------------------------------------

    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };
    int numTokens = sizeof(words) / sizeof(words[0]);

    // Clock for timing gpu 
    auto startCPU = system_clock::now();
    for (const char* word : words)
    {
        int occurrences = calc_token_occurrences(data, word);
        std::cout << "Number of occurrences of \"" << word << "\": " << occurrences << std::endl;
    }
    auto endCPU = system_clock::now();
    auto total = endCPU - startCPU;
    total_ns_CPU += total.count();

    // Writes CPU time to file
    CPUdata_unop << total.count() << endl;

    cout << "CPU total: " << total_ns_CPU << " ns (nano-seconds)" << endl;

    std::vector<int> tokenOffsets(numTokens);
    std::vector<int> tokenLengths(numTokens);
    int totalTokenLen = 0;

    for (int i = 0; i < numTokens; ++i) {
        tokenOffsets[i] = totalTokenLen;
        tokenLengths[i] = strlen(words[i]);
        totalTokenLen += tokenLengths[i];
    }

    std::vector<char> tokens(totalTokenLen);
    for (int i = 0; i < numTokens; ++i) {
        memcpy(&tokens[tokenOffsets[i]], words[i], tokenLengths[i]);
    }

    std::vector<int> results(numTokens, 0);

    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    clGetPlatformIDs(1, &platformId, nullptr);
    clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, nullptr);
    context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(context, deviceId, 0, nullptr);
    program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);
    kernel = clCreateKernel(program, "calcTokenOccurrences", nullptr);

    cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize * sizeof(char), data.data(), nullptr);
    cl_mem d_results = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, numTokens * sizeof(int), results.data(), nullptr);
    cl_mem d_tokens = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, totalTokenLen * sizeof(char), tokens.data(), nullptr);
    cl_mem d_tokenOffsets = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numTokens * sizeof(int), tokenOffsets.data(), nullptr);
    cl_mem d_tokenLengths = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numTokens * sizeof(int), tokenLengths.data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_data);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_results);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_tokens);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_tokenOffsets);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_tokenLengths);
    clSetKernelArg(kernel, 5, sizeof(int), &dataSize);
    clSetKernelArg(kernel, 6, sizeof(int), &numTokens);

    // start GPU timer
    auto startGPU = system_clock::now();

    size_t globalSize = dataSize;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, d_results, CL_TRUE, 0, numTokens * sizeof(int), results.data(), 0, nullptr, nullptr);

    // End and calculate GPU timer
    auto endGPU = system_clock::now();
    auto total_gpu = endGPU - startGPU;
    total_ns_GPU += total_gpu.count();

    clReleaseMemObject(d_data);
    clReleaseMemObject(d_results);
    clReleaseMemObject(d_tokens);
    clReleaseMemObject(d_tokenOffsets);
    clReleaseMemObject(d_tokenLengths);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Writes gpu time to file
    GPUdata_unoptimized << total_gpu.count() << endl;

    // Output the number of occurrences for each token
    for (int i = 0; i < numTokens; ++i) {
        std::cout << "Number of occurrences of \"" << words[i] << "\": " << results[i] << std::endl;
    }

    // Output the number of occurrences for each token
    cout << "GPU total: " << total_ns_GPU << " ns (nano-seconds)" << endl;

    return 0;
}
