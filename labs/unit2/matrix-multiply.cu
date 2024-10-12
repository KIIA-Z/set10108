#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include "gpuErrchk.h"

#define N 512

using namespace std;

//constexpr size_t ELEMENTS = 2048;

__global__ void matrixMultiplyKernal(float* A, float* B, float* C, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0f;

	// Calculate result of one element of matrix C
	if (row < n && col < n) {
		for (int k = 0; k < n; ++k) {
			sum += A[row * n + k] * B[k * n + col];
		}
		C[row * n + col] = sum;
		printf("C[%d][%d] = %f \n", row, col, sum);
	}
}

void matrixMultiply(float* A, float* B, float* C, int n) {
	float* d_A, * d_B, * d_C;
	size_t size = n * n * sizeof(float);

	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_C, size);

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
	matrixMultiplyKernal << <gridSize, blockSize >> > (d_A, d_B, d_C, n);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}




void printMatrix (float* matrix, int n) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			cout << matrix[i * n + j] << " ";
		}
		cout << endl;
	}
}


int main(int argc, char **argv)
{
	//Mutiply vectors 

	float* X, * Y, * Z;

	X = (float*)malloc(N * N * sizeof(float));
	Y = (float*)malloc(N * N * sizeof(float));
	Z = (float*)malloc(N * N * sizeof(float));

	if (X == nullptr || Y == nullptr || Z == nullptr) {
		cerr << "memory allocation failed" << endl;
	}
	
	for (int i = 0; i < N * N; ++i) {
		X[i] = static_cast<float>(rand()) / RAND_MAX;
		Y[i] = static_cast<float>(rand()) / RAND_MAX;
	}

	//matrixMultiply(X, Y, Z, N);


	int n = 2;
	float A[4] = { 1,2,3,4 };
	float B[4] = { 5,6,7,8 };
	float C[4];

	matrixMultiply(A, B, C, n);

	cout << "\nMatrix A:" << endl;
	printMatrix(A, n);
	cout << "Matrix B:" << endl;
	printMatrix(B, n);
	cout << "Matrix C:" << endl;
	printMatrix(C, n);
	cout << "Expected matrix: {19,22,43,50}" << endl;
	
	cout << "Matrix Z:" << endl;
	printMatrix(C, n);
	cout << "Z[0] = " << Z[0] << endl;


	cout << "Finished" << endl;

	free(X);
	free(Y);
	free(Z);

	return 0;
}
