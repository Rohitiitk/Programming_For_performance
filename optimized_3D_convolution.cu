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

const uint64_t N = (1 << 8); // Matrix width
const uint64_t SIZE_IN_BYTES_MATRIX = N * N * N * sizeof(float);
#define M 5 // Convolution filter width
#define TILE_WIDTH 4 // Output tile width
#define BLOCK_WIDTH (TILE_WIDTH + M - 1) // Block width

__host__ __device__ bool is_valid_3D(const int i, const int j, const int k,  const uint64_t N){
	return 0 <= i && i < N && 0 <= j && j < N && 0 <= k && k < N;
}

__constant__ float d_filter[M][M][M];

__global__ void kernel3D(const float *d_a, float *d_b){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	int dep_o = blockIdx.z * TILE_WIDTH + tz; // Output depth
	int row_o = blockIdx.y * TILE_WIDTH + ty; // Output row
	int col_o = blockIdx.x * TILE_WIDTH + tx; // Outut column
	
	int dep_i = dep_o - M / 2; // Input depth
	int row_i = row_o - M / 2; // Input row
	int col_i = col_o - M / 2; // Input column

	__shared__ float temp[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

	if(is_valid_3D(dep_i, row_i, col_i, N)){
		temp[tz][ty][tx] = d_a[dep_i * N * N + row_i * N + col_i];
	}
	else{
		temp[tz][ty][tx] = 0;
	}
	__syncthreads();

	if(is_valid_3D(tz, ty, tx, TILE_WIDTH) && is_valid_3D(dep_o, row_o, col_o, N)){
		float sum = 0;
		for(int i = 0; i < M; i++){
			for(int j = 0; j < M; j++){
				for(int k = 0; k < M; k++){
					sum += temp[tz + i][ty + j][tx + k] * d_filter[i][j][k];
				}
			}
		}
		d_b[dep_o * N * N + row_o * N + col_o] = sum / (M * M * M);
	}
}

__host__ void check_result_3D(const float* w_ref, const float* w_opt) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      for (uint64_t k = 0; k < N; k++) {
        double this_diff =
            w_ref[i * N * N + j * N + k] - w_opt[i * N * N + j * N + k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
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

void print3D(const float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        cout << A[i * N * N + j * N + k] << "\t";
      }
      cout << "\n";
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

void init_matrix_3D(float *a,  const uint64_t N){
		for(int i = 0; i < N; i++){
				for(int j = 0; j < N; j++){
						for(int k = 0; k < N; k++){
								a[i * N * N + j * N + k] = rand() % 100;
						}
				}
		}
}

void calculate_ref_3D(const float *a, float *b, const float *filter){
	float sum;
	for(int i = 0; i < N; i++){
			for(int j = 0; j < N; j++){
				for(int k = 0; k < N; k++){
					sum = 0;		
					for(int di = -M / 2; di <= M / 2; di++){
						for(int dj = -M / 2; dj <= M / 2; dj++){
							for(int dk = -M / 2; dk <= M / 2; dk++){
								if(!is_valid_3D(i + di, j + dj, k + dk, N)){
										continue;
								}
								sum += a[(i + di) * N * N + (j + dj) * N + (k + dk)] * filter[(M / 2 + di) * M * M + (M / 2 + dj) * M + (M / 2 + dk)];
							}
						}
					}
					b[i * N * N + j * N + k] = sum / (M * M * M); 
				}
			}
	}
}

void convolution_3D(){
	int SIZE_IN_BYTES_FILTER = M * M * M * sizeof(float);

  float *a = NULL, *b_ref = NULL, *b = NULL, *filter = NULL;
  a = (float*)malloc(SIZE_IN_BYTES_MATRIX);
  b_ref = (float*)malloc(SIZE_IN_BYTES_MATRIX);
  b = (float*)malloc(SIZE_IN_BYTES_MATRIX);
  filter = (float*)malloc(SIZE_IN_BYTES_FILTER);

	init_matrix_3D(a, N);
	init_matrix_3D(filter, M);

	double clkbegin = rtclock();
	calculate_ref_3D(a, b_ref, filter);
	double clkend = rtclock();
	double cpu_time = clkend - clkbegin;
	cout << "Convolution 3D time on CPU: " << cpu_time * 1000 << " msec " << endl;

  float *d_a = NULL, *d_b = NULL;
	cudaCheckError(cudaMalloc(&d_a, SIZE_IN_BYTES_MATRIX));
	cudaCheckError(cudaMalloc(&d_b, SIZE_IN_BYTES_MATRIX));

	cudaCheckError(cudaMemcpy(d_a, a, SIZE_IN_BYTES_MATRIX, cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpyToSymbol(d_filter, filter, SIZE_IN_BYTES_FILTER));

	dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 numBlocks(N / TILE_WIDTH, N/ TILE_WIDTH, N / TILE_WIDTH);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
	kernel3D<<<numBlocks, threadsPerBlock>>>(d_a, d_b);
	cudaCheckError(cudaPeekAtLastError());
	cudaEventRecord(end, 0);
	cudaCheckError(cudaMemcpy(b, d_b, SIZE_IN_BYTES_MATRIX, cudaMemcpyDeviceToHost));	
	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	check_result_3D(b_ref, b);
	cout << "Convolution 3D time on GPU: " << kernel_time << " msec " << endl;
	
	free(a);
	free(b);
	free(filter);

	cudaFree(d_a);
	cudaFree(d_b);
}

int main() {
  srand(time(NULL));
 
	convolution_3D();

  return EXIT_SUCCESS;
}
