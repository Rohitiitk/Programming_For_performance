#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>

#define THRESHOLD (std::numeric_limits<float>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

const uint64_t N = (1 << 13); // Matrix width
const uint64_t SIZE_IN_BYTES_MATRIX = N * N * sizeof(float);
#define M 5 // Convolution filter width
#define TILE_WIDTH 16 // Output tile width
#define BLOCK_WIDTH (TILE_WIDTH + M - 1) // Block width

__host__ __device__ bool is_valid_2D(const int i, const int j, const int N){
	return 0 <= i && i < N && 0 <= j && j < N;
}

__constant__ float d_filter[M][M];

__global__ void kernel2D(const float *d_a, float *d_b) {
	__shared__ float temp[BLOCK_WIDTH][BLOCK_WIDTH];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row_o = blockIdx.y * TILE_WIDTH + ty; // Output row
	int col_o = blockIdx.x * TILE_WIDTH + tx; // Output column
	
	int row_i = row_o - M / 2; // Input row
	int col_i = col_o - M / 2; // Input column

	if(is_valid_2D(row_i, col_i, N)){
		temp[ty][tx] = d_a[row_i * N + col_i];
	}
	else{
		temp[ty][tx] = 0;
	}
	__syncthreads();

	if(is_valid_2D(ty, tx, TILE_WIDTH) && is_valid_2D(row_o, col_o, N)){
		float sum = 0;
		for(int i = 0; i < M; i++){
			for(int j = 0; j < M; j++){
				sum += temp[ty + i][tx + j] * d_filter[i][j];
			}
		}
		d_b[row_o * N + col_o] = sum / (M * M);
	}
}

__host__ void check_result_2D(const float* w_ref, const float* w_opt) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
        double this_diff =
            w_ref[i * N + j] - w_opt[i * N + j];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print2D(const float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      cout << A[i * N + j] << "\t";
    }
    cout << "\n";
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void init_matrix_2D(float *a, const int N){
		for(int i = 0; i < N; i++){
				for(int j = 0; j < N; j++){
					a[i * N + j] = rand() % 100;
				}
		}
}

void calculate_ref_2D(const float *a, float *b, const float *filter){
	float sum;
	for(int i = 0; i < N; i++){
			for(int j = 0; j < N; j++){
				sum = 0;		
				for(int di = -M / 2; di <= M / 2; di++){
					for(int dj = -M / 2; dj <= M / 2; dj++){
							if(!is_valid_2D(i + di, j + dj, N)){
									continue;
							}
							sum += a[(i + di) * N  + (j + dj)] * filter[(M / 2 + di) * M  + (M / 2 + dj)];
					}
				}
				b[i * N + j] = sum / (M * M); 
			}
	}
}

void convolution_2D(){
	int SIZE_IN_BYTES_FILTER = M * M * sizeof(float);

  float *a = NULL, *b_ref = NULL, *b = NULL, *filter = NULL;
  a = (float*)malloc(SIZE_IN_BYTES_MATRIX);
  b_ref = (float*)malloc(SIZE_IN_BYTES_MATRIX);
  b = (float*)malloc(SIZE_IN_BYTES_MATRIX);
  filter = (float*)malloc(SIZE_IN_BYTES_FILTER);

	init_matrix_2D(a, N);
	init_matrix_2D(filter, M);

	double clkbegin = rtclock();
	calculate_ref_2D(a, b_ref, filter);
	double clkend = rtclock();
	double cpu_time = clkend - clkbegin;
	cout << "Convolution 2D time on CPU: " << cpu_time * 1000 << " msec " << endl;

  float *d_a = NULL, *d_b = NULL;
	cudaCheckError(cudaMalloc(&d_a, SIZE_IN_BYTES_MATRIX));
	cudaCheckError(cudaMalloc(&d_b, SIZE_IN_BYTES_MATRIX));

	cudaCheckError(cudaMemcpy(d_a, a, SIZE_IN_BYTES_MATRIX, cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpyToSymbol(d_filter, filter, SIZE_IN_BYTES_FILTER));

	dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 numBlocks(N / TILE_WIDTH, N / TILE_WIDTH);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
	kernel2D<<<numBlocks, threadsPerBlock>>>(d_a, d_b);
	cudaCheckError(cudaPeekAtLastError());
	cudaEventRecord(end, 0);
	cudaCheckError(cudaMemcpy(b, d_b, SIZE_IN_BYTES_MATRIX, cudaMemcpyDeviceToHost));	
	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	check_result_2D(b_ref, b);
	cout << "Convolution 2D time on GPU: " << kernel_time << " msec " << endl;

	free(a);
	free(b);
	free(filter);

	cudaFree(d_a);
	cudaFree(d_b);
}

int main() {
  srand(time(NULL));
 
	convolution_2D();

  return EXIT_SUCCESS;
}
