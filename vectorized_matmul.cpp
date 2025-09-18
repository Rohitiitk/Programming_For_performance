#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <x86intrin.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const static float EPSILON = std::numeric_limits<float>::epsilon();

#define N (1024)
#define ALIGN 64

void matmul_seq_aligned(float** A, float** B, float** C) {
  __builtin_assume_aligned(A, ALIGN);
  __builtin_assume_aligned(B, ALIGN);
  __builtin_assume_aligned(C, ALIGN);
  float sum = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

void matmul_seq_unaligned(float** A, float** B, float** C) {
  float sum = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

void matmul_sse4_aligned(float** A, float** B, float** C) {
  	__builtin_assume_aligned(A, ALIGN);
  	__builtin_assume_aligned(B, ALIGN);
  	__builtin_assume_aligned(C, ALIGN);
	__m128 rA, rB, rSum, rSum_0, rSum_1, rSum_2, rSum_3;
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j += 4){
			rSum = _mm_setzero_ps();
			for(int k = 0; k < N; k += 4){
				rA = _mm_load_ps1(&A[i][k]);
				rB = _mm_load_ps(&B[k][j]);
				rSum_0 = _mm_mul_ps(rA, rB);

				rA = _mm_load_ps1(&A[i][k + 1]);
				rB = _mm_load_ps(&B[k + 1][j]);
				rSum_1 = _mm_mul_ps(rA, rB);
				
				rA = _mm_load_ps1(&A[i][k + 2]);
				rB = _mm_load_ps(&B[k + 2][j]);
				rSum_2 = _mm_mul_ps(rA, rB);
				
				rA = _mm_load_ps1(&A[i][k + 3]);
				rB = _mm_load_ps(&B[k + 3][j]);
				rSum_3 = _mm_mul_ps(rA, rB);

				rSum = _mm_add_ps(rSum, rSum_0);
				rSum = _mm_add_ps(rSum, rSum_1);
				rSum = _mm_add_ps(rSum, rSum_2);
				rSum = _mm_add_ps(rSum, rSum_3);
			}
			_mm_store_ps(&C[i][j], rSum);
		}
	}
}

void matmul_sse4_unaligned(float** A, float** B, float** C) {
	__m128 rA, rB, rSum, rSum_0, rSum_1, rSum_2, rSum_3;
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j += 4){
			rSum = _mm_setzero_ps();
			for(int k = 0; k < N; k += 4){
				rA = _mm_set_ps1(A[i][k]); // _mm_loadu_ps1 doesn't exist. That's why I am using _mm_set_ps1.
				rB = _mm_loadu_ps(&B[k][j]);
				rSum_0 = _mm_mul_ps(rA, rB);

				rA = _mm_set_ps1(A[i][k + 1]);
				rB = _mm_loadu_ps(&B[k + 1][j]);
				rSum_1 = _mm_mul_ps(rA, rB);
				
				rA = _mm_set_ps1(A[i][k + 2]);
				rB = _mm_loadu_ps(&B[k + 2][j]);
				rSum_2 = _mm_mul_ps(rA, rB);
				
				rA = _mm_set_ps1(A[i][k + 3]);
				rB = _mm_loadu_ps(&B[k + 3][j]);
				rSum_3 = _mm_mul_ps(rA, rB);

				rSum = _mm_add_ps(rSum, rSum_0);
				rSum = _mm_add_ps(rSum, rSum_1);
				rSum = _mm_add_ps(rSum, rSum_2);
				rSum = _mm_add_ps(rSum, rSum_3);
			}
			_mm_storeu_ps(&C[i][j], rSum);
		}
	}
}

void matmul_avx2_aligned(float** A, float** B, float** C) {
  	__builtin_assume_aligned(A, ALIGN);
  	__builtin_assume_aligned(B, ALIGN);
  	__builtin_assume_aligned(C, ALIGN);
	__m256 rA, rB, rSum, rSum_0, rSum_1, rSum_2, rSum_3, rSum_4, rSum_5, rSum_6, rSum_7;
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j += 8){
			rSum = _mm256_setzero_ps();
			for(int k = 0; k < N; k += 8){
				rA = _mm256_set1_ps(A[i][k]); // _mm256_load_ps1 doesn't exist. That's why I am using _mm256_set1_ps.
				rB = _mm256_load_ps(&B[k][j]);
				rSum_0 = _mm256_mul_ps(rA, rB);

				rA = _mm256_set1_ps(A[i][k + 1]);
				rB = _mm256_load_ps(&B[k + 1][j]);
				rSum_1 = _mm256_mul_ps(rA, rB);
				
				rA = _mm256_set1_ps(A[i][k + 2]);
				rB = _mm256_load_ps(&B[k + 2][j]);
				rSum_2 = _mm256_mul_ps(rA, rB);
				
				rA = _mm256_set1_ps(A[i][k + 3]);
				rB = _mm256_load_ps(&B[k + 3][j]);
				rSum_3 = _mm256_mul_ps(rA, rB);

				rA = _mm256_set1_ps(A[i][k + 4]);
				rB = _mm256_load_ps(&B[k + 4][j]);
				rSum_4 = _mm256_mul_ps(rA, rB);
				
				rA = _mm256_set1_ps(A[i][k + 5]);
				rB = _mm256_load_ps(&B[k + 5][j]);
				rSum_5 = _mm256_mul_ps(rA, rB);

				rA = _mm256_set1_ps(A[i][k + 6]);
				rB = _mm256_load_ps(&B[k + 6][j]);
				rSum_6 = _mm256_mul_ps(rA, rB);

				rA = _mm256_set1_ps(A[i][k + 7]);
				rB = _mm256_load_ps(&B[k + 7][j]);
				rSum_7 = _mm256_mul_ps(rA, rB);

				rSum = _mm256_add_ps(rSum, rSum_0);
				rSum = _mm256_add_ps(rSum, rSum_1);
				rSum = _mm256_add_ps(rSum, rSum_2);
				rSum = _mm256_add_ps(rSum, rSum_3);
				rSum = _mm256_add_ps(rSum, rSum_4);
				rSum = _mm256_add_ps(rSum, rSum_5);
				rSum = _mm256_add_ps(rSum, rSum_6);
				rSum = _mm256_add_ps(rSum, rSum_7);
			}
			_mm256_store_ps(&C[i][j], rSum);
		}
	}
}

void matmul_avx2_unaligned(float** A, float** B, float** C) {
	__m256 rA, rB, rSum, rSum_0, rSum_1, rSum_2, rSum_3, rSum_4, rSum_5, rSum_6, rSum_7;
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j += 8){
			rSum = _mm256_setzero_ps();
			for(int k = 0; k < N; k += 8){
				rA = _mm256_set1_ps(A[i][k]);
				rB = _mm256_loadu_ps(&B[k][j]);
				rSum_0 = _mm256_mul_ps(rA, rB);

				rA = _mm256_set1_ps(A[i][k + 1]);
				rB = _mm256_loadu_ps(&B[k + 1][j]);
				rSum_1 = _mm256_mul_ps(rA, rB);
				
				rA = _mm256_set1_ps(A[i][k + 2]);
				rB = _mm256_loadu_ps(&B[k + 2][j]);
				rSum_2 = _mm256_mul_ps(rA, rB);
				
				rA = _mm256_set1_ps(A[i][k + 3]);
				rB = _mm256_loadu_ps(&B[k + 3][j]);
				rSum_3 = _mm256_mul_ps(rA, rB);

				rA = _mm256_set1_ps(A[i][k + 4]);
				rB = _mm256_loadu_ps(&B[k + 4][j]);
				rSum_4 = _mm256_mul_ps(rA, rB);
				
				rA = _mm256_set1_ps(A[i][k + 5]);
				rB = _mm256_loadu_ps(&B[k + 5][j]);
				rSum_5 = _mm256_mul_ps(rA, rB);

				rA = _mm256_set1_ps(A[i][k + 6]);
				rB = _mm256_loadu_ps(&B[k + 6][j]);
				rSum_6 = _mm256_mul_ps(rA, rB);

				rA = _mm256_set1_ps(A[i][k + 7]);
				rB = _mm256_loadu_ps(&B[k + 7][j]);
				rSum_7 = _mm256_mul_ps(rA, rB);

				rSum = _mm256_add_ps(rSum, rSum_0);
				rSum = _mm256_add_ps(rSum, rSum_1);
				rSum = _mm256_add_ps(rSum, rSum_2);
				rSum = _mm256_add_ps(rSum, rSum_3);
				rSum = _mm256_add_ps(rSum, rSum_4);
				rSum = _mm256_add_ps(rSum, rSum_5);
				rSum = _mm256_add_ps(rSum, rSum_6);
				rSum = _mm256_add_ps(rSum, rSum_7);
			}
			_mm256_storeu_ps(&C[i][j], rSum); // C[i][j] may not be 32-byte aligned. That's why I am using unaligned store.
		}
	}
}

void check_result(float** w_ref, float** w_opt) {
  float maxdiff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > EPSILON) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << EPSILON
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

int main() {
  float** A_aligned = (float**)(aligned_alloc(ALIGN, N * sizeof(float*)));
  float** B_aligned = (float**)(aligned_alloc(ALIGN, N * sizeof(float*)));
  float** C_seq_aligned = (float**)(aligned_alloc(ALIGN, N * sizeof(float*)));
  float** C_sse4_aligned = (float**)(aligned_alloc(ALIGN, N * sizeof(float*)));
  float** C_avx2_aligned = (float**)(aligned_alloc(ALIGN, N * sizeof(float*)));
  for (int i = 0; i < N; i++) {
    A_aligned[i] = (float*)(aligned_alloc(ALIGN, N * sizeof(float)));
    B_aligned[i] = (float*)(aligned_alloc(ALIGN, N * sizeof(float)));
    C_seq_aligned[i] = (float*)(aligned_alloc(ALIGN, N * sizeof(float)));
    C_sse4_aligned[i] = (float*)(aligned_alloc(ALIGN, N * sizeof(float)));
    C_avx2_aligned[i] = (float*)(aligned_alloc(ALIGN, N * sizeof(float)));
  }

  float** A_unaligned = (float**)((uintptr_t)(aligned_alloc(ALIGN, (N + 1) * sizeof(float*))) + 1);
  float** B_unaligned = (float**)((uintptr_t)(aligned_alloc(ALIGN, (N + 1) * sizeof(float*))) + 1);
  float** C_seq_unaligned = (float**)((uintptr_t)(aligned_alloc(ALIGN, (N + 1) * sizeof(float*))) + 1);
  float** C_sse4_unaligned = (float**)((uintptr_t)(aligned_alloc(ALIGN, (N + 1) * sizeof(float*))) + 1);
  float** C_avx2_unaligned = (float**)((uintptr_t)(aligned_alloc(ALIGN, (N + 1) * sizeof(float*))) + 1);
  for (int i = 0; i < N; i++) {
    A_unaligned[i] = (float*)((uintptr_t)(aligned_alloc(ALIGN, (N + 1) * sizeof(float))) + 1);
    B_unaligned[i] = (float*)((uintptr_t)(aligned_alloc(ALIGN, (N + 1) * sizeof(float))) + 1);
    C_seq_unaligned[i] = (float*)((uintptr_t)(aligned_alloc(ALIGN, (N + 1) * sizeof(float))) + 1);
    C_sse4_unaligned[i] = (float*)((uintptr_t)(aligned_alloc(ALIGN, (N + 1) * sizeof(float))) + 1);
    C_avx2_unaligned[i] = (float*)((uintptr_t)(aligned_alloc(ALIGN, (N + 1) * sizeof(float))) + 1);
  }
  
  // initialize arrays
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_aligned[i][j] = 0.1F;
      A_unaligned[i][j] = 0.1F;
      B_aligned[i][j] = 0.2F;
      B_unaligned[i][j] = 0.2F;
	  C_seq_aligned[i][j] = 0.0F;
	  C_seq_unaligned[i][j] = 0.0F;
      C_sse4_aligned[i][j] = 0.0F;
      C_sse4_unaligned[i][j] = 0.0F;
      C_avx2_aligned[i][j] = 0.0F;
      C_avx2_unaligned[i][j] = 0.0F;
    }
  }

  HRTimer start = HR::now();
  matmul_seq_aligned(A_aligned, B_aligned, C_seq_aligned);
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul seq aligned time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_seq_unaligned(A_unaligned, B_unaligned, C_seq_unaligned);
  end = HR::now();
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul seq unaligned time: " << duration << " ms" << endl;
  
  start = HR::now();
  matmul_sse4_aligned(A_aligned, B_aligned, C_sse4_aligned);
  end = HR::now();
  check_result(C_seq_aligned, C_sse4_aligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4 aligned time: " << duration << " ms" << endl;
	
  start = HR::now();
  matmul_sse4_unaligned(A_aligned, B_aligned, C_sse4_unaligned);
  end = HR::now();
  check_result(C_seq_aligned, C_sse4_unaligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4 unaligned time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_avx2_aligned(A_aligned, B_aligned, C_avx2_aligned);
  end = HR::now();
  check_result(C_seq_aligned, C_avx2_aligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2 aligned time: " << duration << " ms" << endl;

  start = HR::now();
  matmul_avx2_unaligned(A_aligned, B_aligned, C_avx2_unaligned);
  end = HR::now();
  check_result(C_seq_aligned, C_avx2_unaligned);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2 unaligned time: " << duration << " ms" << endl;
  
  return EXIT_SUCCESS;
}
