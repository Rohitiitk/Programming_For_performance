#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <tuple>
#include <numeric>
#include <sstream>
#include <functional>
#include <papi.h>

using std::cout;
using std::endl;
using uint64 = uint64_t;
using Clock = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;

// Problem sizes
#define INP_H (1 << 7)   // 128
#define INP_W (1 << 7)   // 128
#define INP_D (1 << 7)   // 128

#define FIL_H 3
#define FIL_W 3
#define FIL_D 3

const int RUNS_PER_CONFIG = 5;

// PAPI counters
struct PapiCounts {
    long_long l1_dcm = 0;
    long_long l1_tcm = 0;
    long_long l2_dcm = 0;
    long_long l2_tcm = 0;
    long_long l3_tcm = 0;
};

// Naive 3D cross-correlation
void cc_3d_naive(const uint64* input,
                 const uint64 kernel[FIL_H][FIL_W][FIL_D],
                 uint64* result,
                 uint64 outH, uint64 outW, uint64 outD) {
  for (uint64 i = 0; i < outH; ++i) {
    for (uint64 j = 0; j < outW; ++j) {
      for (uint64 k = 0; k < outD; ++k) {
        uint64 sum = 0;
        for (uint64 ki = 0; ki < FIL_H; ++ki) {
          for (uint64 kj = 0; kj < FIL_W; ++kj) {
            for (uint64 kk = 0; kk < FIL_D; ++kk) {
              uint64 in_idx = (i + ki) * (uint64)INP_W * (uint64)INP_D
                            + (j + kj) * (uint64)INP_D
                            + (k + kk);
              sum += input[in_idx] * kernel[ki][kj][kk];
            }
          }
        }
        uint64 out_idx = i * outW * outD + j * outD + k;
        result[out_idx] += sum;
      }
    }
  }
}

// Blocked 3D cross-correlation
void cc_3d_blocked(const uint64* input,
                   const uint64 kernel[FIL_H][FIL_W][FIL_D],
                   uint64* result,
                   uint64 outH, uint64 outW, uint64 outD,
                   uint64 TILE_H, uint64 TILE_W, uint64 TILE_D) {
  for (uint64 ii = 0; ii < outH; ii += TILE_H) {
    for (uint64 jj = 0; jj < outW; jj += TILE_W) {
      for (uint64 kk = 0; kk < outD; kk += TILE_D) {
        uint64 iMax = std::min(ii + TILE_H, outH);
        uint64 jMax = std::min(jj + TILE_W, outW);
        uint64 kMax = std::min(kk + TILE_D, outD);

        for (uint64 i = ii; i < iMax; ++i) {
          for (uint64 j = jj; j < jMax; ++j) {
            for (uint64 k = kk; k < kMax; ++k) {
              uint64 sum = 0;
              for (uint64 ki = 0; ki < FIL_H; ++ki) {
                for (uint64 kj = 0; kj < FIL_W; ++kj) {
                  for (uint64 kk2 = 0; kk2 < FIL_D; ++kk2) {
                    uint64 in_idx = (i + ki) * (uint64)INP_W * (uint64)INP_D
                                  + (j + kj) * (uint64)INP_D
                                  + (k + kk2);
                    sum += input[in_idx] * kernel[ki][kj][kk2];
                  }
                }
              }
              uint64 out_idx = i * outW * outD + j * outD + k;
              result[out_idx] += sum;
            }
          }
        }
      }
    }
  }
}

// Timer wrappers
double time_kernel_naive(const uint64* input,
                         const uint64 kernel[FIL_H][FIL_W][FIL_D],
                         uint64* result,
                         uint64 outH, uint64 outW, uint64 outD) {
  std::fill_n(result, outH * outW * outD, 0);
  auto t0 = Clock::now();
  cc_3d_naive(input, kernel, result, outH, outW, outD);
  auto t1 = Clock::now();
  return Seconds(t1 - t0).count();
}

double time_kernel_blocked(const uint64* input,
                           const uint64 kernel[FIL_H][FIL_W][FIL_D],
                           uint64* result,
                           uint64 outH, uint64 outW, uint64 outD,
                           uint64 th, uint64 tw, uint64 td) {
  std::fill_n(result, outH * outW * outD, 0);
  auto t0 = Clock::now();
  cc_3d_blocked(input, kernel, result, outH, outW, outD, th, tw, td);
  auto t1 = Clock::now();
  return Seconds(t1 - t0).count();
}

// PAPI measurement wrapper
PapiCounts papi_run_and_count(std::function<void()> kernel_fun) {
  int EventSet = PAPI_NULL;
  PAPI_create_eventset(&EventSet);

  std::vector<int> events = {
    PAPI_L1_DCM,
    PAPI_L1_TCM,
    PAPI_L2_DCM,
    PAPI_L2_TCM,
    PAPI_L3_TCM
  };

  for (int ev : events) PAPI_add_event(EventSet, ev);
  std::vector<long_long> values(events.size(), 0);

  PAPI_start(EventSet);
  kernel_fun();
  PAPI_stop(EventSet, values.data());
  PAPI_cleanup_eventset(EventSet);
  PAPI_destroy_eventset(&EventSet);

  PapiCounts res;
  res.l1_dcm = values[0];
  res.l1_tcm = values[1];
  res.l2_dcm = values[2];
  res.l2_tcm = values[3];
  res.l3_tcm = values[4];
  return res;
}

std::vector<uint64> default_candidates() {
  return {2,4,8,16,32,64};
}

std::string papi_to_string(const PapiCounts &p) {
  std::ostringstream oss;
  oss << "L1_DCM=" << p.l1_dcm
      << "  L1_TCM=" << p.l1_tcm
      << "  L2_DCM=" << p.l2_dcm
      << "  L2_TCM=" << p.l2_tcm
      << "  L3_TCM=" << p.l3_tcm;
  return oss.str();
}


// helper: run with PAPI and get average time + cache stats
PapiCounts run_with_papi(std::function<void()> kernel, int runs = 5) {
    PapiCounts avg{};
    double total_time = 0.0;

    for (int r = 0; r < runs; r++) {
        long long values[5] = {0};
        int events[5] = { PAPI_L1_DCM, PAPI_L1_TCM, PAPI_L2_DCM, PAPI_L2_TCM, PAPI_L3_TCM };
        int EventSet = PAPI_NULL;
        PAPI_create_eventset(&EventSet);
        PAPI_add_events(EventSet, events, 5);
        PAPI_start(EventSet);

        auto start = std::chrono::high_resolution_clock::now();
        kernel();
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<double, std::milli>(end - start).count();

        PAPI_stop(EventSet, values);
        PAPI_cleanup_eventset(EventSet);
        PAPI_destroy_eventset(&EventSet);

        avg.l1_dcm += values[0];
        avg.l1_tcm += values[1];
        avg.l2_dcm += values[2];
        avg.l2_tcm += values[3];
        avg.l3_tcm += values[4];
    }

    // take averages
    avg.l1_dcm /= runs;
    avg.l1_tcm /= runs;
    avg.l2_dcm /= runs;
    avg.l2_tcm /= runs;
    avg.l3_tcm /= runs;

    cout << "Average time (ms): " << (total_time / runs) << endl;
    cout << "Average L1 DCM: " << avg.l1_dcm << endl;
    cout << "Average L1 TCM: " << avg.l1_tcm << endl;
    cout << "Average L2 DCM: " << avg.l2_dcm << endl;
    cout << "Average L2 TCM: " << avg.l2_tcm << endl;
    cout << "Average L3 TCM: " << avg.l3_tcm << endl;

    return avg;
}


int main() {
  PAPI_library_init(PAPI_VER_CURRENT);

  uint64 outH = INP_H - FIL_H + 1;
  uint64 outW = INP_W - FIL_W + 1;
  uint64 outD = INP_D - FIL_D + 1;

  cout << "Input: " << INP_H << "x" << INP_W << "x" << INP_D << "\n";
  cout << "Filter: " << FIL_H << "x" << FIL_W << "x" << FIL_D << "\n";
  cout << "Output: " << outH << "x" << outW << "x" << outD << "\n\n";

  size_t input_size = (size_t)INP_H * INP_W * INP_D;
  uint64* input; posix_memalign((void**)&input, 64, input_size * sizeof(uint64));
  std::fill_n(input, input_size, (uint64)1);

  uint64 filter[FIL_H][FIL_W][FIL_D];
  for (int i=0;i<FIL_H;++i)
    for (int j=0;j<FIL_W;++j)
      for (int k=0;k<FIL_D;++k)
        filter[i][j][k] = 2 + (i+j+k)%3;

  size_t result_size = (size_t)outH * outW * outD;
  uint64* result; posix_memalign((void**)&result, 64, result_size * sizeof(uint64));

  // === Naive baseline ===
  cout << "=== Naive ===\n";
  double naive_avg = 0;
  for (int r=0;r<RUNS_PER_CONFIG;++r) {
    double t = time_kernel_naive(input, filter, result, outH, outW, outD);
    naive_avg += t;
    cout << " run " << (r+1) << " time=" << t << " s\n";
  }
  naive_avg /= RUNS_PER_CONFIG;
  cout << "Naive average time = " << naive_avg << " s\n\n";

  // === Blocked autotuning ===
  auto candidates = default_candidates();
  double best_time = std::numeric_limits<double>::max();
  std::tuple<int,int,int> best_cfg;

  cout << "=== Blocked ===\n";
  for (auto th : candidates) {
    for (auto tw : candidates) {
      for (auto td : candidates) {
        double avg = 0;
        for (int r=0;r<RUNS_PER_CONFIG;++r) {
          double t = time_kernel_blocked(input, filter, result, outH, outW, outD, th, tw, td);
          avg += t;
        }
        avg /= RUNS_PER_CONFIG;
        cout << " tile(" << th << "," << tw << "," << td << ") avg time=" << avg << " s\n";

        if (avg < best_time) {
          best_time = avg;
          best_cfg = {th,tw,td};
        }
      }
    }
  }

  auto [th,tw,td] = best_cfg;
  cout << "\nBest blocked config = (" << th << "," << tw << "," << td << ")\n";
  cout << "Best time = " << best_time << " s\n";
  cout << "Speedup over naive = " << (naive_avg / best_time) << "x\n";


cout << "\n--- Naive Kernel (5 runs) ---\n";
auto naive_stats = run_with_papi([&](){
    naive_kernel(input, result, input_size);
});

cout << "\n--- Best Blocked Kernel (5 runs) ---\n";
auto best_stats = run_with_papi([&](){
    blocked_kernel(input, result, input_size, best_th, best_tw, best_td);
});


  free(input);
  free(result);
  PAPI_shutdown();
  return 0;
}


