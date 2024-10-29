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

constexpr int ELEMENTS = 5096;
//constexpr int ELEMENTS = 2048;
//constexpr int ELEMENTS = 1024;
constexpr std::size_t DATA_SIZE = sizeof(int) * ELEMENTS;

std::vector<char> read_file(const char* filename)
{
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);

    // Check if the file opened successfully
    if (!file) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return {};
    }

    // Move the file cursor to the end of the file to get its size
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();

    // Return the file cursor to the beginning of the file
    file.seekg(0, std::ios::beg);

    // Create a vector of the same size as the file to hold the content
    std::vector<char> buffer(fileSize);

    // Read the entire file into the vector
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Error: Could not read the file content." << std::endl;
        return {};
    }

    // Close the file
    file.close();

    // Output the number of bytes read
    std::cout << "Successfully read " << buffer.size() << " bytes from the file." << std::endl;

    // convert to lowercase
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), [](char c) { return std::tolower(c); });

    return buffer;
}

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
    //const char * filepath = "dataset/shakespeare.txt";

    //initilise timers at zero for cpu and gpu runtime testing
    uint64_t total_ns_GPU = 0;
    uint64_t total_ns_CPU = 0;

    int user_choice;
    std::vector<char> data;
    int exit = 0;

    //std::cout << std::filesystem::current_path() << std::endl;
    
    // a switch case surrounded by a do whoile loop that allows the user to chose which file to read to file
    std::cout << "1 for Beowulf" << std::endl;
    std::cout << "2 for Crime and Punishment" << std::endl;
    std::cout << "3 for Edgar Allan Poe" << std::endl;
    std::cout << "4 for Pride and Prejudice." << std::endl;
    std::cout << "5 for Shakespeare" << std::endl;
    std::cout << "Please enter a number between 1-5: ";
    do {
        std::cin >> user_choice;
        switch (user_choice) {
        case 1:
            data = read_file("dataset/beowulf.txt");
            cout << "beowulf.txt file has been read!" << endl;
            cout << " " << endl;
            exit = 1;
            break;
        case 2:
            data = read_file("dataset/crime_and_punishment.txt");
            cout << "crime_and_punishment.txt file has been read!" << endl;
            cout << " " << endl;
            exit = 1;
            break;
        case 3:
            data = read_file("dataset/edgar_allan_poe.txt");
            cout << "edgar_allan_poe.txt file has been read!" << endl;
            cout << " " << endl;
            exit = 1;
            break;
        case 4:
            data = read_file("dataset/pride_and_prejudice.txt");
            cout << "pride_and_prejudice.txt file has been read!" << endl;
            cout << " " << endl;
            exit = 1;
            break;
        case 5:
            data = read_file("dataset/shakespeare.txt");
            cout << "shakespeare.txt file has been read!" << endl;
            cout << " " << endl;
                exit = 1;
            break;
        default:
            cout << "please only enter a number between 1-5: " << endl;
            cout << " " << endl;

        }
    } while (exit != 1);
    
    //-------------------CPU/Original code Area---------------------------------------------
    

    const char* words[] = { "sword", "fire", "death", "love", "hate", "the", "man", "woman" };
    int numTokens = sizeof(words) / sizeof(words[0]);

    //clock for timing gpu 
    auto startCPU = system_clock::now();

    for (const char* word : words)
    {
        int occurrences = calc_token_occurrences(data, word);
        std::cout << "Number of occurrences of \"" << word << "\": " << occurrences << std::endl;
    }
    auto endCPU = system_clock::now();

    auto total = endCPU - startCPU;
    total_ns_CPU += total.count();

    cout << "CPU total: " << total_ns_CPU << "ns (nano-seconds)" << endl;

    //-------------------GPU/OpenCL code Area---------------------------------------------

    // Calculate the total length and offsets of all tokens
    int totalTokenLen = 0;
    std::vector<int> tokenOffsets(numTokens);
    std::vector<int> tokenLengths(numTokens);
    for (int i = 0; i < numTokens; ++i) {
        tokenOffsets[i] = totalTokenLen;
        tokenLengths[i] = strlen(words[i]);
        totalTokenLen += tokenLengths[i];
    }
    // Flatten the tokens array into a single buffer
    std::vector<char> tokens;
    for (int i = 0; i < numTokens; ++i) {
        tokens.insert(tokens.end(), words[i], words[i] + tokenLengths[i]);
    }

    // Read the data from a file
    //std::vector<char> data = read_file("C:/Users/kia/source/repos/set10108/labs/cw1/dataset/shakespeare.txt");
    int dataSize = data.size();
    // Initialize result
    std::vector<int> results(numTokens, 0); // One result per token

    try {
    // Get the platforms
    std::vector<Platform> platforms;
    Platform::get(&platforms);

    // Assume only one platform.  Get GPU devices.
    std::vector<Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // Just to test, print out device 0 name
    cout << devices[0].getInfo<CL_DEVICE_NAME>() << endl;

    // Create a context with these devices
    Context context(devices);

    // Create a command queue for device 0
    CommandQueue queue(context, devices[0]);

    // Create the buffers
    Buffer buf_data(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data.size() * sizeof(char), data.data());
    Buffer buf_results(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, results.size() * sizeof(int), results.data());
    Buffer buf_tokens(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tokens.size() * sizeof(char), tokens.data());
    Buffer buf_tokenOffsets(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tokenOffsets.size() * sizeof(int), tokenOffsets.data());
    Buffer buf_tokenLength(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tokenLengths.size() * sizeof(int), tokenLengths.data());


    // Copy data to the GPU
    queue.enqueueWriteBuffer(buf_data, CL_TRUE, 0, data.size() * sizeof(char), data.data());
    queue.enqueueWriteBuffer(buf_results, CL_TRUE, 0, results.size() * sizeof(int), results.data());
    queue.enqueueWriteBuffer(buf_tokens, CL_TRUE, 0, tokens.size() * sizeof(char), tokens.data());
    queue.enqueueWriteBuffer(buf_tokenOffsets, CL_TRUE, 0, tokenOffsets.size() * sizeof(int), tokenOffsets.data());
    queue.enqueueWriteBuffer(buf_tokenLength, CL_TRUE, 0, tokenLengths.size() * sizeof(int), tokenLengths.data());

    // Read in kernel source
    ifstream file("calc-token-occurrences.cl");
    string code(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));

    // Create program
    Program::Sources source{ code };
    Program program(context, source);
    
    // Build program for devices 
    cl_int buildErr = program.build(devices);
    if (buildErr != CL_SUCCESS) {
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
        std::cerr << "Error building program: " << buildErr << std::endl;
        std::cerr << "Build log:\n" << buildLog << std::endl;
    }

    // Create the kernel
    Kernel calc_token_occurrences_kernel(program, "calcTokenOccurrences");

    // Set kernel arguments
    calc_token_occurrences_kernel.setArg(0, buf_data);
    calc_token_occurrences_kernel.setArg(1, buf_results);
    calc_token_occurrences_kernel.setArg(2, buf_tokens);
    calc_token_occurrences_kernel.setArg(3, buf_tokenOffsets);
    calc_token_occurrences_kernel.setArg(4, buf_tokenLength);
    calc_token_occurrences_kernel.setArg(5, dataSize);
    calc_token_occurrences_kernel.setArg(6, numTokens);


    //test that shows the max work size for gpu (debug stuff)
    size_t maxWorkGroupSize;
    devices[0].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
    std::cout << "Max work group size: " << maxWorkGroupSize << std::endl;

    //calulates value for local size using the devices max work size 
    size_t globalSize = dataSize;
    size_t localSize = (dataSize / (1024));
    if (localSize > maxWorkGroupSize) {
        localSize = maxWorkGroupSize;  // Adjust to fit within the limit
    }
    // Ensure local size fits evenly into the global size
    if (globalSize % localSize != 0) {
        globalSize = (globalSize / localSize) * localSize;  // Adjust global size
    }

    std::cout << "WorkGroupSize: " << dataSize << std::endl;

    //start GPU timer
    auto startGPU = system_clock::now();
    
    //set gobal and local work sizes to caculated values
    NDRange global(globalSize);
    NDRange local(localSize);

    // Execute kernel
    queue.enqueueNDRangeKernel(calc_token_occurrences_kernel, NullRange, global, local);
    queue.finish();

    // Copy result back
    queue.enqueueReadBuffer(buf_results, CL_TRUE, 0, results.size() * sizeof(int), results.data());

     for (int i = 0; i < numTokens; ++i) {
        std::cout << "Number of occurrences of \"" << words[i] << "\": " << results[i] << std::endl;
    }

     //end and calculate GPU timer
     auto endGPU = system_clock::now();
     auto total_gpu = endGPU - startGPU;
     total_ns_GPU += total_gpu.count();

     // Output the number of occurrences for each token
     cout << "GPU total: " << total_ns_GPU << "ns (nano-seconds)" << endl;
}
catch (Error error)
{
    cout << error.what() << "(" << error.err() << ")" << endl;
}
    
    // Clean up memory storage etc
    //clReleaseMemObject(buf_data);
    //clReleaseMemObject(buf_results);
    //clReleaseMemObject(buf_tokens);
    //clReleaseMemObject(buf_tokenOffsets);
    //clReleaseMemObject(buf_tokenLength);
    //clReleaseKernel(calc_token_occurrences_kernel);
    //clReleaseProgram(program);
    //clReleaseCommandQueue(queue);
    //clReleaseContext(context);
   
    return 0;
}
