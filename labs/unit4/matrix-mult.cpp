#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <CL/cl.hpp>

using namespace std;
using namespace cl;

constexpr int ELEMENTS = 2048;
constexpr std::size_t DATA_SIZE = sizeof(int) * ELEMENTS;


int main(int argc, char* argv[])
{
        std::vector<float> h_A, h_B, h_C; // matrices
        int Mdim, Ndim, Pdim; // A[N][P],B[P][M],C[N][M]
        int i, err;
        int szA, szB, szC; // num elements in each matrix
        double start_time, run_time; // timing data
        cl::Program program;

        Ndim = Pdim = Mdim = ORDER;
        szA = Ndim * Pdim;
        szB = Pdim * Mdim;
        szC = Ndim * Mdim;
        h_A = std::vector<float>(szA);
        h_B = std::vector<float>(szB);
        h_C = std::vector<float>(szC);

        initmat(Mdim, Ndim, Pdim, h_A, h_B, h_C);

        // Compile for first kernel to setup program
        program = cl::Program(C_elem_KernelSource, true);
        Context context(CL_DEVICE_TYPE_DEFAULT);
        cl::CommandQueue queue(context);
        std::vector<Device> devices =
            context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::string s = device.getInfo<CL_DEVICE_NAME>();
        std::cout << "\nUsing OpenCL Device" << s << "\n";
        // Setup the buffers, initialize matrices,
        // and write them into global memory
        initmat(Mdim, Ndim, Pdim, h_A, h_B, h_C);
        cl::Buffer d_a(context, h_A.begin(), h_A.end(), true);
        cl::Buffer d_b(context, h_B.begin(), h_B.end(), true);
        cl::Buffer d_c = cl::Buffer(context,
            CL_MEM_WRITE_ONLY,
            sizeof(float) * szC);

        cl::make_kernel<int, int, int,cl::Buffer, cl::Buffer, cl::Buffer>rowcol(program, "mmul");

        zero_mat(Ndim, Mdim, h_C);
        start_time = wtime();

        krow(cl::EnqueueArgs(queue,
            cl::NDRange(Ndim),
            cl::NDRange(ORDER / 16)),
            Ndim, Mdim, Pdim, a_in, b_in, c_out);


        cl::copy(queue, d_c, h_C.begin(), h_C.end());

        run_time = wtime() - start_time;
        results(Mdim, Ndim, Pdim, h_C, run_time);
        
}

}