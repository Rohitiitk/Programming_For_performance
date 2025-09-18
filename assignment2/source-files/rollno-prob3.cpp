// #include <atomic>
// #include <cassert>
// #include <chrono>
// #include <cstdint>
// #include <cstdlib>
// #include <cstring>
// #include <iostream>
// #include <pthread.h>
// #include <vector>
// #include <algorithm>
// #include <thread>

// using std::cerr;
// using std::cout;
// using std::endl;

// using HR = std::chrono::high_resolution_clock;
// using HRTimer = HR::time_point;
// using std::chrono::duration_cast;
// using std::chrono::microseconds;
// using std::chrono::milliseconds;

// #define N (1e2) //change
// #define NUM_THREADS (64)

// // Shared variables
// uint64_t var1 = 0, var2 = (N * NUM_THREADS + 1);

// // Abstract base class
// class LockBase {
// public:
//   // Pure virtual function
//   virtual void acquire(uint16_t tid) = 0;
//   virtual void release(uint16_t tid) = 0;
// };

// typedef struct thr_args {
//   uint16_t m_id;
//   LockBase* m_lock;
// } ThreadArgs;

// /** Use pthread mutex to implement lock routines */
// class PthreadMutex : public LockBase {
// public:
//   void acquire(uint16_t tid) override { pthread_mutex_lock(&lock); }
//   void release(uint16_t tid) override { pthread_mutex_unlock(&lock); }

// private:
//   pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
// };

// // /** Filter Lock Implementation */
// // class FilterLock : public LockBase {
// // private:
// //   static const int MAX_THREADS = NUM_THREADS;
// //   std::atomic<int> level[MAX_THREADS];
// //   std::atomic<int> victim[MAX_THREADS];
// //   int n_threads;

// // public:
// //   FilterLock(int num_threads = NUM_THREADS) : n_threads(num_threads) {
// //     for (int i = 0; i < MAX_THREADS; i++) {
// //       level[i].store(0);
// //       victim[i].store(0);
// //     }
// //   }

// //   void acquire(uint16_t tid) override {
// //     for (int i = 1; i < n_threads; i++) {
// //       level[tid].store(i);
// //       victim[i].store(tid);
      
// //       // Spin while conflicts exist
// //       bool wait = true;
// //       while (wait) {
// //         wait = false;
// //         for (int k = 0; k < n_threads; k++) {
// //           if (k != tid && level[k].load() >= i && victim[i].load() == tid) {
// //             wait = true;
// //             break;
// //           }
// //         }
// //       }
// //     }
// //   }

// //   void release(uint16_t tid) override {
// //     level[tid].store(0);
// //   }

// //   ~FilterLock() {}
// // };

// // /** Bakery Lock Implementation */
// // class BakeryLock : public LockBase {
// // private:
// //   static const int MAX_THREADS = NUM_THREADS;
// //   alignas(64) std::atomic<bool> choosing[MAX_THREADS];
// //   alignas(64) std::atomic<int> number[MAX_THREADS];
// //   int n_threads;

// // public:
// //   BakeryLock(int num_threads = NUM_THREADS) : n_threads(num_threads) {
// //     for (int i = 0; i < MAX_THREADS; i++) {
// //       choosing[i].store(false);
// //       number[i].store(0);
// //     }
// //   }

// //   void acquire(uint16_t tid) override {
// //     choosing[tid].store(true);
    
// //     // Find max ticket number
// //     int max_num = 0;
// //     for (int i = 0; i < n_threads; i++) {
// //       int num = number[i].load();
// //       if (num > max_num) {
// //         max_num = num;
// //       }
// //     }
    
// //     number[tid].store(max_num + 1);
// //     choosing[tid].store(false);
    
// //     // Wait for all threads with smaller tickets
// //     for (int i = 0; i < n_threads; i++) {
// //       if (i == tid) continue;
      
// //       // Wait while thread i is choosing
// //       while (choosing[i].load());
      
// //       // Wait while thread i has priority
// //       while (true) {
// //         int num_i = number[i].load();
// //         if (num_i == 0) break;
        
// //         int num_tid = number[tid].load();
// //         if (num_i > num_tid) break;
// //         if (num_i == num_tid && i > tid) break;
// //       }
// //     }
// //   }

// //   void release(uint16_t tid) override {
// //     number[tid].store(0);
// //   }

// //   ~BakeryLock() {}
// // };

// // /** Spin Lock Implementation */
// // class SpinLock : public LockBase {
// // private:
// //   std::atomic_flag lock_flag = ATOMIC_FLAG_INIT;

// // public:
// //   void acquire(uint16_t tid) override {
// //     while (lock_flag.test_and_set(std::memory_order_acquire)) {
// //       // Spin
// //       // Busy-wait until the lock is released
// //     }
// //   }

// //   void release(uint16_t tid) override {
// //     lock_flag.clear(std::memory_order_release);
// //   }

// //   SpinLock() {}
// //   ~SpinLock() {}
// // };

// // /** Ticket Lock Implementation */
// // class TicketLock : public LockBase {
// // private:
// //   alignas(64) std::atomic<uint64_t> next_ticket;
// //   alignas(64) std::atomic<uint64_t> now_serving;

// // public:
// //   TicketLock() : next_ticket(0), now_serving(0) {}

// //   void acquire(uint16_t tid) override {
// //     uint64_t my_ticket = next_ticket.fetch_add(1, std::memory_order_relaxed);
    
// //     while (now_serving.load(std::memory_order_acquire) != my_ticket) {
// //       // Spin with backoff
// //       while (now_serving.load(std::memory_order_relaxed) != my_ticket);
// //     }
// //   }

// //   void release(uint16_t tid) override {
// //     now_serving.fetch_add(1, std::memory_order_release);
// //   }

// //   ~TicketLock() {}
// // };

// // /** Array-based Queue Lock Implementation */
// // class ArrayQLock : public LockBase {
// // private:
// //   static const int MAX_THREADS = NUM_THREADS;
// //   alignas(64) std::atomic<bool> flags[MAX_THREADS];
// //   alignas(64) std::atomic<int> tail;
// //   alignas(64) int my_slot[MAX_THREADS];
// //   int n_threads;

// // public:
// //   ArrayQLock(int num_threads = NUM_THREADS) : n_threads(num_threads), tail(0) {
// //     for (int i = 0; i < MAX_THREADS; i++) {
// //       flags[i].store(false);
// //       my_slot[i] = -1;
// //     }
// //     flags[0].store(true);  // Initially unlocked
// //   }

// //   void acquire(uint16_t tid) override {
// //     int slot = tail.fetch_add(1, std::memory_order_relaxed) % n_threads;
// //     my_slot[tid] = slot;
    
// //     while (!flags[slot].load(std::memory_order_acquire)) {
// //       // Spin
// //     }
    
// //     flags[slot].store(false, std::memory_order_relaxed);
// //   }

// //   void release(uint16_t tid) override {
// //     int slot = my_slot[tid];
// //     int next_slot = (slot + 1) % n_threads;
// //     flags[next_slot].store(true, std::memory_order_release);
// //   }

// //   ~ArrayQLock() {}
// // };


// // // class FilterLock : public LockBase {
// // // public:
// // //   void acquire(uint16_t tid) override {}
// // //   void release(uint16_t tid) override {}

// // //   FilterLock() {}
// // //   ~FilterLock() {}
// // // };



// /* Helper: pause hint for x86 */
// static inline void cpu_relax() {
// #if defined(__i386__) || defined(__x86_64__)
//   __builtin_ia32_pause();
// #else
//   // fallback
//   std::this_thread::yield();
// #endif
// }

// class FilterLock : public LockBase {
// public:
//   FilterLock() : n(NUM_THREADS), level(n), victim(n) {
//     for (int i = 0; i < n; ++i) {
//       level[i].store(0, std::memory_order_relaxed);
//     }
//     for (int L = 0; L < n; ++L) {
//       victim[L].store(-1, std::memory_order_relaxed);
//     }
//   }

//   void acquire(uint16_t tid) override {
//     for (int L = 1; L < n; ++L) {
//       level[tid].store(L, std::memory_order_release);
//       victim[L].store((int)tid, std::memory_order_release);
//       // wait while some other thread has level >= L and victim[L] == tid
//       // bool spinning = true;
//       // while (spinning) {
//       //   spinning = false;
//       //   if (victim[L].load(std::memory_order_acquire) != (int)tid) {
//       //     // victim changed, proceed to re-evaluate
//       //     spinning = false;
//       //     continue;
//       //   }
//       //   for (int k = 0; k < n; ++k) {
//       //     if (k == (int)tid) continue;
//       //     if (level[k].load(std::memory_order_acquire) >= L &&
//       //         victim[L].load(std::memory_order_acquire) == (int)tid) {
//       //       spinning = true;
//       //       cpu_relax();
//       //       break;
//       //     }
//       //   }
//       // }

//             // wait while there exists a thread k != tid with level[k] >= L and victim[L] == tid
//       while (true) {
//         bool someone_at_level = false;
//         for (int k = 0; k < n; ++k) {
//           if (k == (int)tid) continue;
//           if (level[k].load(std::memory_order_acquire) >= L &&
//               victim[L].load(std::memory_order_acquire) == (int)tid) {
//             someone_at_level = true;
//             break;
//           }
//         }
//         if (!someone_at_level) break;
//         cpu_relax();
//       }

//     }
//   }

//   void release(uint16_t tid) override {
//     level[tid].store(0, std::memory_order_release);
//   }

// private:
//   const int n;
//   std::vector<std::atomic<int>> level;   // level per thread
//   std::vector<std::atomic<int>> victim;  // victim per level
// };

// /* -------------------- Bakery Lock -------------------- */
// /*
//   Bakery lock: choosing[tid], number[tid]
//   acquire: choosing[tid]=true, number[tid]=1+max(number[]), choosing[tid]=false
//            wait for every j: while choosing[j] ; then while number[j] != 0 && (number[j], j) < (number[tid], tid)
//   release: number[tid] = 0
// */
// class BakeryLock : public LockBase {
// public:
//   BakeryLock() : n(NUM_THREADS), choosing(n), number(n) {
//     for (int i = 0; i < n; ++i) {
//       choosing[i].store(false, std::memory_order_relaxed);
//       number[i].store(0ULL, std::memory_order_relaxed);
//     }
//   }

//   void acquire(uint16_t tid) override {
//     choosing[tid].store(true, std::memory_order_release);

//     // get max ticket
//     uint64_t maxnum = 0;
//     for (int j = 0; j < n; ++j) {
//       uint64_t v = number[j].load(std::memory_order_acquire);
//       if (v > maxnum) maxnum = v;
//     }
//     number[tid].store(maxnum + 1, std::memory_order_release);
//     choosing[tid].store(false, std::memory_order_release);

//     uint64_t mynum = number[tid].load(std::memory_order_acquire);
//     for (int j = 0; j < n; ++j) {
//       if (j == (int)tid) continue;
//       // wait if j is choosing
//       while (choosing[j].load(std::memory_order_acquire)) {
//         cpu_relax();
//       }
//       // wait while j's ticket is non-zero and (number[j], j) < (mynum, tid)
//       while (true) {
//         uint64_t nj = number[j].load(std::memory_order_acquire);
//         if (nj == 0) break;
//         // if nj < mynum OR (nj == mynum && j < tid) then wait
//         if (nj < mynum || (nj == mynum && j < (int)tid)) {
//           cpu_relax();
//           continue;
//         } else {
//           break;
//         }
//       }
//     }
//   }

//   void release(uint16_t tid) override {
//     number[tid].store(0ULL, std::memory_order_release);
//   }

// private:
//   const int n;
//   std::vector<std::atomic<bool>> choosing;
//   std::vector<std::atomic<uint64_t>> number;
// };

// /* -------------------- Spin Lock -------------------- */
// /*
//   Simple test-and-set using atomic_flag
// */
// class SpinLock : public LockBase {
// public:
//   SpinLock() { flag.clear(); }
//   void acquire(uint16_t /*tid*/) override {
//     while (flag.test_and_set(std::memory_order_acquire)) {
//       cpu_relax();
//     }
//   }
//   void release(uint16_t /*tid*/) override { flag.clear(std::memory_order_release); }

// private:
//   std::atomic_flag flag = ATOMIC_FLAG_INIT;
// };

// /* -------------------- Ticket Lock -------------------- */
// /*
//   FIFO ticket lock with next (ticket dispenser) and serving
// */
// class TicketLock : public LockBase {
// public:
//   TicketLock() : next(0), serving(0) {}

//   void acquire(uint16_t /*tid*/) override {
//     uint64_t my = next.fetch_add(1, std::memory_order_relaxed);
//     while (serving.load(std::memory_order_acquire) != my) {
//       cpu_relax();
//     }
//   }

//   void release(uint16_t /*tid*/) override { serving.fetch_add(1, std::memory_order_release); }

// private:
//   std::atomic<uint64_t> next;
//   std::atomic<uint64_t> serving;
// };

// /* -------------------- Array-based Queue Lock (Anderson) -------------------- */
// /*
//   Anderson lock: allocate flags[maxThreads]; tail FAA; each thread gets slot = tail++ % size
//   spin on flags[slot]; on release set flags[slot]=false and flags[(slot+1)%size]=true
//   We keep per-thread slot in slot_by_tid[tid].
// */
// class ArrayQLock : public LockBase {
// public:
//   ArrayQLock() : size(NUM_THREADS), tail(0), flags(size), slot_by_tid(size) {
//     for (int i = 0; i < size; ++i) {
//       flags[i].store(false, std::memory_order_relaxed);
//       slot_by_tid[i].store(-1, std::memory_order_relaxed);
//     }
//     // first slot available initially
//     flags[0].store(true, std::memory_order_relaxed);
//   }

//   void acquire(uint16_t tid) override {
//     uint64_t pos = tail.fetch_add(1, std::memory_order_relaxed);
//     int slot = int(pos % size);
//     slot_by_tid[tid].store(slot, std::memory_order_release);
//     // spin on own slot flag
//     while (!flags[slot].load(std::memory_order_acquire)) {
//       cpu_relax();
//     }
//   }

//   void release(uint16_t tid) override {
//     int myslot = slot_by_tid[tid].load(std::memory_order_acquire);
//     // mark myslot false, make next true
//     flags[myslot].store(false, std::memory_order_release);
//     int next = (myslot + 1) % size;
//     flags[next].store(true, std::memory_order_release);
//   }

// private:
//   const int size;
//   std::atomic<uint64_t> tail;
//   std::vector<std::atomic<bool>> flags;
//   std::vector<std::atomic<int>> slot_by_tid; // indexed by thread id
// };

// // // #include <atomic>
// // // #include <vector>
// // // #include <cstdint>

// // // // Helper for spinning
// // // inline void cpu_relax() { asm volatile("pause" ::: "memory"); }

// // // constexpr int NUM_THREADS = 8; // adjust as needed

// // // class LockBase {
// // // public:
// // //     virtual void acquire(uint16_t tid) = 0;
// // //     virtual void release(uint16_t tid) = 0;
// // //     virtual ~LockBase() = default;
// // // };

// // // /* -------------------- Filter Lock -------------------- */
// // // class FilterLock : public LockBase {
// // // public:
// // //     FilterLock() : n(NUM_THREADS), level(n), victim(n) {
// // //         for (int i = 0; i < n; ++i) {
// // //             level[i].val.store(0, std::memory_order_relaxed);
// // //             victim[i].val.store(-1, std::memory_order_relaxed);
// // //         }
// // //     }

// // //     void acquire(uint16_t tid) override {
// // //         for (int L = 1; L < n; ++L) {
// // //             level[tid].val.store(L, std::memory_order_release);
// // //             victim[L].val.store((int)tid, std::memory_order_release);

// // //             while (true) {
// // //                 bool someone_at_level = false;
// // //                 for (int k = 0; k < n; ++k) {
// // //                     if (k == (int)tid) continue;
// // //                     int vk = victim[L].val.load(std::memory_order_acquire);
// // //                     int lk = level[k].val.load(std::memory_order_acquire);
// // //                     if (lk >= L && vk == (int)tid) {
// // //                         someone_at_level = true;
// // //                         break;
// // //                     }
// // //                 }
// // //                 if (!someone_at_level) break;
// // //                 cpu_relax();
// // //             }
// // //         }
// // //     }

// // //     void release(uint16_t tid) override {
// // //         level[tid].val.store(0, std::memory_order_release);
// // //     }

// // // private:
// // //     struct alignas(64) PaddedInt { std::atomic<int> val; };
// // //     const int n;
// // //     std::vector<PaddedInt> level;   // per-thread level
// // //     std::vector<PaddedInt> victim;  // per-level victim
// // // };

// // // /* -------------------- Bakery Lock -------------------- */
// // // class BakeryLock : public LockBase {
// // // public:
// // //     BakeryLock() : n(NUM_THREADS), choosing(n), number(n) {
// // //         for (int i = 0; i < n; ++i) {
// // //             choosing[i].val.store(false, std::memory_order_relaxed);
// // //             number[i].val.store(0ULL, std::memory_order_relaxed);
// // //         }
// // //     }

// // //     void acquire(uint16_t tid) override {
// // //         choosing[tid].val.store(true, std::memory_order_release);

// // //         uint64_t maxnum = 0;
// // //         for (int j = 0; j < n; ++j) {
// // //             uint64_t v = number[j].val.load(std::memory_order_acquire);
// // //             if (v > maxnum) maxnum = v;
// // //         }
// // //         number[tid].val.store(maxnum + 1, std::memory_order_release);
// // //         choosing[tid].val.store(false, std::memory_order_release);

// // //         uint64_t mynum = number[tid].val.load(std::memory_order_acquire);
// // //         for (int j = 0; j < n; ++j) {
// // //             if (j == (int)tid) continue;
// // //             while (choosing[j].val.load(std::memory_order_acquire)) cpu_relax();
// // //             while (true) {
// // //                 uint64_t nj = number[j].val.load(std::memory_order_acquire);
// // //                 if (nj == 0) break;
// // //                 if (nj < mynum || (nj == mynum && j < (int)tid)) {
// // //                     cpu_relax();
// // //                     continue;
// // //                 } else break;
// // //             }
// // //         }
// // //     }

// // //     void release(uint16_t tid) override {
// // //         number[tid].val.store(0ULL, std::memory_order_release);
// // //     }

// // // private:
// // //     struct alignas(64) PaddedBool { std::atomic<bool> val; };
// // //     struct alignas(64) PaddedUint64 { std::atomic<uint64_t> val; };
// // //     const int n;
// // //     std::vector<PaddedBool> choosing;
// // //     std::vector<PaddedUint64> number;
// // // };

// // // /* -------------------- Spin Lock -------------------- */
// // // class SpinLock : public LockBase {
// // // public:
// // //     SpinLock() { flag.clear(); }

// // //     void acquire(uint16_t /*tid*/) override {
// // //         while (flag.test_and_set(std::memory_order_acquire)) cpu_relax();
// // //     }
// // //     void release(uint16_t /*tid*/) override { flag.clear(std::memory_order_release); }

// // // private:
// // //     std::atomic_flag flag = ATOMIC_FLAG_INIT;
// // // };

// // // /* -------------------- Ticket Lock -------------------- */
// // // class TicketLock : public LockBase {
// // // public:
// // //     TicketLock() : next(0), serving(0) {}

// // //     void acquire(uint16_t /*tid*/) override {
// // //         uint64_t my = next.fetch_add(1, std::memory_order_relaxed);
// // //         while (serving.load(std::memory_order_acquire) != my) cpu_relax();
// // //     }

// // //     void release(uint16_t /*tid*/) override { serving.fetch_add(1, std::memory_order_release); }

// // // private:
// // //     std::atomic<uint64_t> next;
// // //     std::atomic<uint64_t> serving;
// // // };

// // // /* -------------------- Array-based Queue Lock (Anderson) -------------------- */
// // // class ArrayQLock : public LockBase {
// // // public:
// // //     ArrayQLock() : size(NUM_THREADS), tail(0), flags(size), slot_by_tid(size) {
// // //         for (int i = 0; i < size; ++i) {
// // //             flags[i].val.store(false, std::memory_order_relaxed);
// // //             slot_by_tid[i].val.store(-1, std::memory_order_relaxed);
// // //         }
// // //         flags[0].val.store(true, std::memory_order_relaxed); // first slot available
// // //     }

// // //     void acquire(uint16_t tid) override {
// // //         uint64_t pos = tail.fetch_add(1, std::memory_order_relaxed);
// // //         int slot = int(pos % size);
// // //         slot_by_tid[tid].val.store(slot, std::memory_order_release);

// // //         while (!flags[slot].val.load(std::memory_order_acquire)) cpu_relax();
// // //     }

// // //     void release(uint16_t tid) override {
// // //         int myslot = slot_by_tid[tid].val.load(std::memory_order_acquire);
// // //         flags[myslot].val.store(false, std::memory_order_release);
// // //         int next = (myslot + 1) % size;
// // //         flags[next].val.store(true, std::memory_order_release);
// // //     }

// // // private:
// // //     struct alignas(64) PaddedBool { std::atomic<bool> val; };
// // //     struct alignas(64) PaddedInt { std::atomic<int> val; };

// // //     const int size;
// // //     std::atomic<uint64_t> tail;
// // //     std::vector<PaddedBool> flags;
// // //     std::vector<PaddedInt> slot_by_tid;
// // // };


// // constexpr size_t CACHELINE_SIZE = 64;

// // /* Implement Filter lock */
// // class FilterLock : public LockBase {
// // public:
// //     void acquire(uint16_t tid) override {
// //         for (int level = 0; level < NUM_THREADS; ++level) {
// //             // Announce thread's arrival at this level
// //             level_[tid].store(level, std::memory_order_relaxed);
// //             // Announce thread's intention to be the victim for this level
// //             victim_[level].store(tid, std::memory_order_relaxed);
            
// //             // Wait for other threads at this level to be "victims"
// //             for (uint16_t k = 0; k < NUM_THREADS; ++k) {
// //                 // Do not wait for self
// //                 if (k == tid) continue;

// //                 // Loop while the other thread k has an equal or higher level
// //                 // and this thread is the designated victim for the current level
// //                 while (level_[k].load(std::memory_order_relaxed) >= level && victim_[level].load(std::memory_order_relaxed) == tid) {
// //                     // spin-wait
// //                 }
// //             }
// //         }
// //     }

// //     void release(uint16_t tid) override {
// //         // A thread exiting the critical section sets its level to -1 (or 0)
// //         // This signifies it is no longer contending for the lock
// //         level_[tid].store(-1, std::memory_order_relaxed);
// //     }

// //     FilterLock() {
// //         level_ = new std::atomic<int>[NUM_THREADS];
// //         victim_ = new std::atomic<uint16_t>[NUM_THREADS];
// //         for (int i = 0; i < NUM_THREADS; ++i) {
// //             level_[i].store(-1);
// //         }
// //     }

// //     ~FilterLock() {
// //         delete[] level_;
// //         delete[] victim_;
// //     }

// // private:
// //     std::atomic<int>* level_;
// //     std::atomic<uint16_t>* victim_;
// // };

// // /* Implement Bakery lock */
// // class BakeryLock : public LockBase {
// // public:
// //     void acquire(uint16_t tid) override {
// //         // Step 1: Choosing a number
// //         choosing_[tid].store(true, std::memory_order_relaxed);
        
// //         // Find the maximum ticket number currently in use
// //         int max_num = 0;
// //         for (uint16_t i = 0; i < NUM_THREADS; ++i) {
// //             int current_num = number_[i].load(std::memory_order_relaxed);
// //             if (current_num > max_num) {
// //                 max_num = current_num;
// //             }
// //         }

// //         // Assign this thread a ticket number one greater than the maximum
// //         number_[tid].store(max_num + 1, std::memory_order_relaxed);
// //         choosing_[tid].store(false, std::memory_order_relaxed);

// //         // Step 2: Waiting for turn
// //         for (uint16_t other = 0; other < NUM_THREADS; ++other) {
// //             if (other == tid) continue;

// //             // Wait until the other thread has finished choosing its number
// //             while (choosing_[other].load(std::memory_order_relaxed)) {
// //                 // spin-wait
// //             }
            
// //             // Wait until the other thread's number is 0 (not in critical section)
// //             // or the current thread's number is smaller
// //             while (number_[other].load(std::memory_order_relaxed) != 0 &&
// //                    (number_[other].load(std::memory_order_relaxed) < number_[tid].load(std::memory_order_relaxed) ||
// //                     (number_[other].load(std::memory_order_relaxed) == number_[tid].load(std::memory_order_relaxed) && other < tid))) {
// //                 // spin-wait
// //             }
// //         }
// //     }

// //     void release(uint16_t tid) override {
// //         // Release the lock by resetting the thread's number to 0
// //         number_[tid].store(0, std::memory_order_relaxed);
// //     }

// //     BakeryLock() {
// //         choosing_ = new std::atomic<bool>[NUM_THREADS];
// //         number_ = new std::atomic<int>[NUM_THREADS];
// //         for (int i = 0; i < NUM_THREADS; ++i) {
// //             choosing_[i].store(false);
// //             number_[i].store(0);
// //         }
// //     }

// //     ~BakeryLock() {
// //         delete[] choosing_;
// //         delete[] number_;
// //     }

// // private:
// //     std::atomic<bool>* choosing_;
// //     std::atomic<int>* number_;
// // };

// // /* Implement Spin lock */
// // class SpinLock : public LockBase {
// // public:
// //     void acquire(uint16_t tid) override {
// //         // Spin-wait until the lock is available and acquire it
// //         while (flag_.test_and_set(std::memory_order_acquire)) {
// //             // The __builtin_ia32_pause() intrinsic is used to
// //             // hint to the processor that the thread is in a spin-wait loop.
// //             // This can improve performance on multi-threaded CPUs by reducing power consumption
// //             // and avoiding a penalty on the memory system.
// //             __builtin_ia32_pause();
// //         }
// //     }
    
// //     void release(uint16_t tid) override {
// //         // Atomically set the flag to false to release the lock
// //         flag_.clear(std::memory_order_release);
// //     }
    
// //     SpinLock() {}
// //     ~SpinLock() {}
// // private:
// //     std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
// // };

// // /* Implement Ticket lock */
// // class TicketLock : public LockBase {
// // public:
// //     void acquire(uint16_t tid) override {
// //         // Atomically get a ticket number and increment the counter
// //         const uint16_t my_ticket = next_ticket_.fetch_add(1, std::memory_order_relaxed);
        
// //         // Spin-wait until the thread's ticket matches the current serving ticket
// //         while (now_serving_.load(std::memory_order_acquire) != my_ticket) {
// //             __builtin_ia32_pause();
// //         }
// //     }
    
// //     void release(uint16_t tid) override {
// //         // Atomically increment the serving ticket to release the lock
// //         const uint16_t next_ticket = now_serving_.load(std::memory_order_relaxed) + 1;
// //         now_serving_.store(next_ticket, std::memory_order_release);
// //     }
    
// //     TicketLock() {}
// //     ~TicketLock() {}
// // private:
// //     // Align variables to different cache lines to prevent false sharing
// //     alignas(CACHELINE_SIZE) std::atomic<uint16_t> next_ticket_ = {0};
// //     alignas(CACHELINE_SIZE) std::atomic<uint16_t> now_serving_ = {0};
// // };

// // /* Implement Array-based Queue lock */
// // class ArrayQLock : public LockBase {
// // public:
// //     void acquire(uint16_t tid) override {
// //         // Atomically get a ticket number and increment the counter, wrapping around the array size
// //         const uint16_t my_ticket = next_ticket_.fetch_add(1, std::memory_order_relaxed) % NUM_THREADS;
        
// //         // Spin-wait until the thread's slot in the array is "true"
// //         while (!flag_[my_ticket].load(std::memory_order_acquire)) {
// //             __builtin_ia32_pause();
// //         }
        
// //         // Mark the slot as "false" for the next thread to enter the queue
// //         flag_[my_ticket].store(false, std::memory_order_relaxed);
// //     }

// //     void release(uint16_t tid) override {
// //         // Calculate the next ticket number
// //         const uint16_t next_ticket = (next_ticket_.load(std::memory_order_relaxed) % NUM_THREADS) - 1;
        
// //         // Signal the next thread in the queue by setting its slot to "true"
// //         flag_[(next_ticket + 1) % NUM_THREADS].store(true, std::memory_order_release);
// //     }

// //     ArrayQLock() {
// //         // Initialize the flag array
// //         flag_ = new std::atomic<bool>[NUM_THREADS];
// //         for (int i = 0; i < NUM_THREADS; ++i) {
// //             flag_[i].store(false);
// //         }
// //         // The first slot is initially true to allow the first thread to acquire the lock
// //         flag_[0].store(true);
// //     }
    
// //     ~ArrayQLock() {
// //         delete[] flag_;
// //     }

// // private:
// //     // A single atomic counter for tickets
// //     alignas(CACHELINE_SIZE) std::atomic<uint16_t> next_ticket_ = {0};
// //     // An array of atomic booleans, each on a separate cache line to prevent false sharing
// //     std::atomic<bool>* flag_;
// // };


// /** Estimate the time taken */
// std::atomic_uint64_t sync_time = 0;

// inline void critical_section() {
//   var1++;
//   var2--;
// }

// /** Sync threads at the start to maximize contention */
// pthread_barrier_t g_barrier;

// void* thrBody(void* arguments) {
//   ThreadArgs* tmp = static_cast<ThreadArgs*>(arguments);
//   if (false) {    // left
//     cout << "Thread id: " << tmp->m_id << " starting\n";
//   }

//   // Wait for all other producer threads to launch before proceeding.
//   pthread_barrier_wait(&g_barrier);

//   HRTimer start = HR::now();
//   for (int i = 0; i < N; i++) {
//     tmp->m_lock->acquire(tmp->m_id);
//     critical_section();
//     tmp->m_lock->release(tmp->m_id);
//   }
//   HRTimer end = HR::now();
//   auto duration = duration_cast<milliseconds>(end - start).count();

//   // A barrier is not required here
//   sync_time.fetch_add(duration);
//   pthread_exit(NULL);
// }

// int main() {
//   int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
//   if (error != 0) {
//     cerr << "Error in barrier init.\n";
//     exit(EXIT_FAILURE);
//   }

//   pthread_attr_t attr;
//   pthread_attr_init(&attr);
//   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

//   pthread_t tid[NUM_THREADS];
//   ThreadArgs args[NUM_THREADS] = {{0}};

//   // Pthread mutex
//   LockBase* lock_obj = new PthreadMutex();
//   uint16_t i = 0;
//   while (i < NUM_THREADS) {
//     args[i].m_id = i;
//     args[i].m_lock = lock_obj;

//     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
//     if (error != 0) {
//       cerr << "\nThread cannot be created : " << strerror(error) << "\n";
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }

//   i = 0;
//   void* status;
//   while (i < NUM_THREADS) {
//     error = pthread_join(tid[i], &status);
//     if (error) {
//       cerr << "ERROR: return code from pthread_join() is " << error << "\n";
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }

//   // assert(var1 == N * NUM_THREADS && var2 == 1);
//   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
//   cout << "Pthread mutex: Time taken (us): " << sync_time << "\n";

//   // Filter lock
//   var1 = 0;
//   var2 = (N * NUM_THREADS + 1);
//   sync_time.store(0);

//   lock_obj = new FilterLock();
//   i = 0;
//   while (i < NUM_THREADS) {
//     args[i].m_id = i;
//     args[i].m_lock = lock_obj;

//     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
//     if (error != 0) {
//       printf("\nThread cannot be created : [%s]", strerror(error));
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }

//   i = 0;
//   while (i < NUM_THREADS) {
//     error = pthread_join(tid[i], &status);
//     if (error) {
//       printf("ERROR: return code from pthread_join() is %d\n", error);
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }

//   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
//   // assert(var1 == N * NUM_THREADS && var2 == 1);
//   cout << "Filter lock: Time taken (us): " << sync_time << "\n";

//   // Bakery lock
//   var1 = 0;
//   var2 = (N * NUM_THREADS + 1);
//   sync_time.store(0);

//   lock_obj = new BakeryLock();
//   i = 0;
//   while (i < NUM_THREADS) {
//     args[i].m_id = i;
//     args[i].m_lock = lock_obj;

//     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
//     if (error != 0) {
//       printf("\nThread cannot be created : [%s]", strerror(error));
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }

//   i = 0;
//   // cout<<"Running Bakery Lock\n";
//   while (i < NUM_THREADS) {
//     error = pthread_join(tid[i], &status);
//     if (error) {
//       printf("ERROR: return code from pthread_join() is %d\n", error);
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }
//   // cout<<"Bakery Loop End\n";

//   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
//   // assert(var1 == N * NUM_THREADS && var2 == 1);
//   cout << "Bakery lock: Time taken (us): " << sync_time << "\n";

//   // Spin lock
//   var1 = 0;
//   var2 = (N * NUM_THREADS + 1);
//   sync_time.store(0);

//   lock_obj = new SpinLock();
//   i = 0;
//   while (i < NUM_THREADS) {
//     args[i].m_id = i;
//     args[i].m_lock = lock_obj;

//     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
//     if (error != 0) {
//       printf("\nThread cannot be created : [%s]", strerror(error));
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }

//   i = 0;
//   while (i < NUM_THREADS) {
//     error = pthread_join(tid[i], &status);
//     if (error) {
//       printf("ERROR: return code from pthread_join() is %d\n", error);
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }

//   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
//   // assert(var1 == N * NUM_THREADS && var2 == 1);
//   cout << "Spin lock: Time taken (us): " << sync_time << "\n";

//   // Ticket lock
//   var1 = 0;
//   var2 = (N * NUM_THREADS + 1);
//   sync_time.store(0);

//   lock_obj = new TicketLock();
//   i = 0;
//   while (i < NUM_THREADS) {
//     args[i].m_id = i;
//     args[i].m_lock = lock_obj;

//     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
//     if (error != 0) {
//       printf("\nThread cannot be created : [%s]", strerror(error));
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }

//   i = 0;
//   while (i < NUM_THREADS) {
//     error = pthread_join(tid[i], &status);
//     if (error) {
//       printf("ERROR: return code from pthread_join() is %d\n", error);
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }

//   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
//   // assert(var1 == N * NUM_THREADS && var2 == 1);
//   cout << "Ticket lock: Time taken (us): " << sync_time << "\n";

//   // Array Q lock
//   var1 = 0;
//   var2 = (N * NUM_THREADS + 1);
//   sync_time.store(0);

//   lock_obj = new ArrayQLock();
//   i = 0;
//   while (i < NUM_THREADS) {
//     args[i].m_id = i;
//     args[i].m_lock = lock_obj;

//     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
//     if (error != 0) {
//       printf("\nThread cannot be created : [%s]", strerror(error));
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }

//   i = 0;
//   while (i < NUM_THREADS) {
//     error = pthread_join(tid[i], &status);
//     if (error) {
//       printf("ERROR: return code from pthread_join() is %d\n", error);
//       exit(EXIT_FAILURE);
//     }
//     i++;
//   }

//   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
//   // assert(var1 == N * NUM_THREADS && var2 == 1);
//   cout << "Array Q lock: Time taken (us): " << sync_time << "\n";

//   pthread_barrier_destroy(&g_barrier);
//   pthread_attr_destroy(&attr);

//   pthread_exit(NULL);
// }


// // // #include <atomic>
// // // #include <cassert>
// // // #include <chrono>
// // // #include <cstdint>
// // // #include <cstdlib>
// // // #include <cstring>
// // // #include <iostream>
// // // #include <pthread.h>
// // // #include <vector>
// // // #include <algorithm>

// // // using std::cerr;
// // // using std::cout;
// // // using std::endl;

// // // using HR = std::chrono::high_resolution_clock;
// // // using HRTimer = HR::time_point;
// // // using std::chrono::duration_cast;
// // // using std::chrono::microseconds;
// // // using std::chrono::milliseconds;

// // // #define N (10000000)          // replaced (1e7) with integer literal for safety
// // // #define NUM_THREADS (8)

// // // // Shared variables
// // // uint64_t var1 = 0, var2 = (uint64_t(N) * NUM_THREADS + 1);

// // // // Abstract base class
// // // class LockBase {
// // // public:
// // //   virtual void acquire(uint16_t tid) = 0;
// // //   virtual void release(uint16_t tid) = 0;
// // //   virtual ~LockBase() = default;
// // // };

// // // typedef struct thr_args {
// // //   uint16_t m_id;
// // //   LockBase* m_lock;
// // // } ThreadArgs;

// // // /** Use pthread mutex to implement lock routines */
// // // class PthreadMutex : public LockBase {
// // // public:
// // //   void acquire(uint16_t /*tid*/) override { pthread_mutex_lock(&lock); }
// // //   void release(uint16_t /*tid*/) override { pthread_mutex_unlock(&lock); }

// // // private:
// // //   pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
// // // };

// // // /* Helper: pause hint for x86 */
// // // static inline void cpu_relax() {
// // // #if defined(__i386__) || defined(__x86_64__)
// // //   __builtin_ia32_pause();
// // // #else
// // //   // fallback
// // //   std::this_thread::yield();
// // // #endif
// // // }

// // // /* -------------------- Filter Lock -------------------- */
// // // /*
// // //   Filter lock: level[tid] in [0..n-1], victim[level] indicates which thread is last to try that level.
// // //   acquire: for L=1..n-1 set level[tid]=L; victim[L]=tid; wait while exists k != tid with level[k] >= L && victim[L]==tid
// // // */
// // // class FilterLock : public LockBase {
// // // public:
// // //   FilterLock() : n(NUM_THREADS), level(n), victim(n) {
// // //     for (int i = 0; i < n; ++i) {
// // //       level[i].store(0, std::memory_order_relaxed);
// // //     }
// // //     for (int L = 0; L < n; ++L) {
// // //       victim[L].store(-1, std::memory_order_relaxed);
// // //     }
// // //   }

// // //   void acquire(uint16_t tid) override {
// // //     for (int L = 1; L < n; ++L) {
// // //       level[tid].store(L, std::memory_order_release);
// // //       victim[L].store((int)tid, std::memory_order_release);
// // //       // wait while some other thread has level >= L and victim[L] == tid
// // //       bool spinning = true;
// // //       while (spinning) {
// // //         spinning = false;
// // //         if (victim[L].load(std::memory_order_acquire) != (int)tid) {
// // //           // victim changed, proceed to re-evaluate
// // //           spinning = false;
// // //           continue;
// // //         }
// // //         for (int k = 0; k < n; ++k) {
// // //           if (k == (int)tid) continue;
// // //           if (level[k].load(std::memory_order_acquire) >= L &&
// // //               victim[L].load(std::memory_order_acquire) == (int)tid) {
// // //             spinning = true;
// // //             cpu_relax();
// // //             break;
// // //           }
// // //         }
// // //       }
// // //     }
// // //   }

// // //   void release(uint16_t tid) override {
// // //     level[tid].store(0, std::memory_order_release);
// // //   }

// // // private:
// // //   const int n;
// // //   std::vector<std::atomic<int>> level;   // level per thread
// // //   std::vector<std::atomic<int>> victim;  // victim per level
// // // };

// // // /* -------------------- Bakery Lock -------------------- */
// // // /*
// // //   Bakery lock: choosing[tid], number[tid]
// // //   acquire: choosing[tid]=true, number[tid]=1+max(number[]), choosing[tid]=false
// // //            wait for every j: while choosing[j] ; then while number[j] != 0 && (number[j], j) < (number[tid], tid)
// // //   release: number[tid] = 0
// // // */
// // // class BakeryLock : public LockBase {
// // // public:
// // //   BakeryLock() : n(NUM_THREADS), choosing(n), number(n) {
// // //     for (int i = 0; i < n; ++i) {
// // //       choosing[i].store(false, std::memory_order_relaxed);
// // //       number[i].store(0ULL, std::memory_order_relaxed);
// // //     }
// // //   }

// // //   void acquire(uint16_t tid) override {
// // //     choosing[tid].store(true, std::memory_order_release);

// // //     // get max ticket
// // //     uint64_t maxnum = 0;
// // //     for (int j = 0; j < n; ++j) {
// // //       uint64_t v = number[j].load(std::memory_order_acquire);
// // //       if (v > maxnum) maxnum = v;
// // //     }
// // //     number[tid].store(maxnum + 1, std::memory_order_release);
// // //     choosing[tid].store(false, std::memory_order_release);

// // //     uint64_t mynum = number[tid].load(std::memory_order_acquire);
// // //     for (int j = 0; j < n; ++j) {
// // //       if (j == (int)tid) continue;
// // //       // wait if j is choosing
// // //       while (choosing[j].load(std::memory_order_acquire)) {
// // //         cpu_relax();
// // //       }
// // //       // wait while j's ticket is non-zero and (number[j], j) < (mynum, tid)
// // //       while (true) {
// // //         uint64_t nj = number[j].load(std::memory_order_acquire);
// // //         if (nj == 0) break;
// // //         // if nj < mynum OR (nj == mynum && j < tid) then wait
// // //         if (nj < mynum || (nj == mynum && j < (int)tid)) {
// // //           cpu_relax();
// // //           continue;
// // //         } else {
// // //           break;
// // //         }
// // //       }
// // //     }
// // //   }

// // //   void release(uint16_t tid) override {
// // //     number[tid].store(0ULL, std::memory_order_release);
// // //   }

// // // private:
// // //   const int n;
// // //   std::vector<std::atomic<bool>> choosing;
// // //   std::vector<std::atomic<uint64_t>> number;
// // // };

// // // /* -------------------- Spin Lock -------------------- */
// // // /*
// // //   Simple test-and-set using atomic_flag
// // // */
// // // class SpinLock : public LockBase {
// // // public:
// // //   SpinLock() { flag.clear(); }
// // //   void acquire(uint16_t /*tid*/) override {
// // //     while (flag.test_and_set(std::memory_order_acquire)) {
// // //       cpu_relax();
// // //     }
// // //   }
// // //   void release(uint16_t /*tid*/) override { flag.clear(std::memory_order_release); }

// // // private:
// // //   std::atomic_flag flag = ATOMIC_FLAG_INIT;
// // // };

// // // /* -------------------- Ticket Lock -------------------- */
// // // /*
// // //   FIFO ticket lock with next (ticket dispenser) and serving
// // // */
// // // class TicketLock : public LockBase {
// // // public:
// // //   TicketLock() : next(0), serving(0) {}

// // //   void acquire(uint16_t /*tid*/) override {
// // //     uint64_t my = next.fetch_add(1, std::memory_order_relaxed);
// // //     while (serving.load(std::memory_order_acquire) != my) {
// // //       cpu_relax();
// // //     }
// // //   }

// // //   void release(uint16_t /*tid*/) override { serving.fetch_add(1, std::memory_order_release); }

// // // private:
// // //   std::atomic<uint64_t> next;
// // //   std::atomic<uint64_t> serving;
// // // };

// // // /* -------------------- Array-based Queue Lock (Anderson) -------------------- */
// // // /*
// // //   Anderson lock: allocate flags[maxThreads]; tail FAA; each thread gets slot = tail++ % size
// // //   spin on flags[slot]; on release set flags[slot]=false and flags[(slot+1)%size]=true
// // //   We keep per-thread slot in slot_by_tid[tid].
// // // */
// // // class ArrayQLock : public LockBase {
// // // public:
// // //   ArrayQLock() : size(NUM_THREADS), tail(0), flags(size), slot_by_tid(size) {
// // //     for (int i = 0; i < size; ++i) {
// // //       flags[i].store(false, std::memory_order_relaxed);
// // //       slot_by_tid[i].store(-1, std::memory_order_relaxed);
// // //     }
// // //     // first slot available initially
// // //     flags[0].store(true, std::memory_order_relaxed);
// // //   }

// // //   void acquire(uint16_t tid) override {
// // //     uint64_t pos = tail.fetch_add(1, std::memory_order_relaxed);
// // //     int slot = int(pos % size);
// // //     slot_by_tid[tid].store(slot, std::memory_order_release);
// // //     // spin on own slot flag
// // //     while (!flags[slot].load(std::memory_order_acquire)) {
// // //       cpu_relax();
// // //     }
// // //   }

// // //   void release(uint16_t tid) override {
// // //     int myslot = slot_by_tid[tid].load(std::memory_order_acquire);
// // //     // mark myslot false, make next true
// // //     flags[myslot].store(false, std::memory_order_release);
// // //     int next = (myslot + 1) % size;
// // //     flags[next].store(true, std::memory_order_release);
// // //   }

// // // private:
// // //   const int size;
// // //   std::atomic<uint64_t> tail;
// // //   std::vector<std::atomic<bool>> flags;
// // //   std::vector<std::atomic<int>> slot_by_tid; // indexed by thread id
// // // };

// // // /** Estimate the time taken */
// // // std::atomic_uint64_t sync_time = 0;

// // // inline void critical_section() {
// // //   var1++;
// // //   var2--;
// // // }

// // // /** Sync threads at the start to maximize contention */
// // // pthread_barrier_t g_barrier;

// // // void* thrBody(void* arguments) {
// // //   ThreadArgs* tmp = static_cast<ThreadArgs*>(arguments);

// // //   // Wait for all other producer threads to launch before proceeding.
// // //   pthread_barrier_wait(&g_barrier);

// // //   HRTimer start = HR::now();
// // //   for (int i = 0; i < N; i++) {
// // //     tmp->m_lock->acquire(tmp->m_id);
// // //     critical_section();
// // //     tmp->m_lock->release(tmp->m_id);
// // //   }
// // //   HRTimer end = HR::now();
// // //   auto duration = duration_cast<milliseconds>(end - start).count();

// // //   sync_time.fetch_add((uint64_t)duration, std::memory_order_relaxed);
// // //   pthread_exit(NULL);
// // // }

// // // int main() {
// // //   int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
// // //   if (error != 0) {
// // //     cerr << "Error in barrier init.\n";
// // //     exit(EXIT_FAILURE);
// // //   }

// // //   pthread_attr_t attr;
// // //   pthread_attr_init(&attr);
// // //   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

// // //   pthread_t tid[NUM_THREADS];
// // //   ThreadArgs args[NUM_THREADS] = {{0}};

// // //   // Pthread mutex
// // //   LockBase* lock_obj = new PthreadMutex();
// // //   uint16_t i = 0;
// // //   while (i < NUM_THREADS) {
// // //     args[i].m_id = i;
// // //     args[i].m_lock = lock_obj;

// // //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// // //     if (error != 0) {
// // //       cerr << "\nThread cannot be created : " << strerror(error) << "\n";
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   i = 0;
// // //   void* status;
// // //   while (i < NUM_THREADS) {
// // //     error = pthread_join(tid[i], &status);
// // //     if (error) {
// // //       cerr << "ERROR: return code from pthread_join() is " << error << "\n";
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   assert(var1 == uint64_t(N) * NUM_THREADS && var2 == 1);
// // //   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
// // //   cout << "Pthread mutex: Time taken (ms): " << sync_time.load() << "\n";

// // //   // Filter lock
// // //   var1 = 0;
// // //   var2 = (uint64_t(N) * NUM_THREADS + 1);
// // //   sync_time.store(0);

// // //   lock_obj = new FilterLock();
// // //   i = 0;
// // //   while (i < NUM_THREADS) {
// // //     args[i].m_id = i;
// // //     args[i].m_lock = lock_obj;

// // //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// // //     if (error != 0) {
// // //       printf("\nThread cannot be created : [%s]", strerror(error));
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   i = 0;
// // //   while (i < NUM_THREADS) {
// // //     error = pthread_join(tid[i], &status);
// // //     if (error != 0) {
// // //       printf("ERROR: return code from pthread_join() is %d\n", error);
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
// // //   cout << "Filter lock: Time taken (ms): " << sync_time.load() << "\n";

// // //   // Bakery lock
// // //   var1 = 0;
// // //   var2 = (uint64_t(N) * NUM_THREADS + 1);
// // //   sync_time.store(0);

// // //   lock_obj = new BakeryLock();
// // //   i = 0;
// // //   while (i < NUM_THREADS) {
// // //     args[i].m_id = i;
// // //     args[i].m_lock = lock_obj;

// // //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// // //     if (error != 0) {
// // //       printf("\nThread cannot be created : [%s]", strerror(error));
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   i = 0;
// // //   while (i < NUM_THREADS) {
// // //     error = pthread_join(tid[i], &status);
// // //     if (error != 0) {
// // //       printf("ERROR: return code from pthread_join() is %d\n", error);
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
// // //   cout << "Bakery lock: Time taken (ms): " << sync_time.load() << "\n";

// // //   // Spin lock
// // //   var1 = 0;
// // //   var2 = (uint64_t(N) * NUM_THREADS + 1);
// // //   sync_time.store(0);

// // //   lock_obj = new SpinLock();
// // //   i = 0;
// // //   while (i < NUM_THREADS) {
// // //     args[i].m_id = i;
// // //     args[i].m_lock = lock_obj;

// // //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// // //     if (error != 0) {
// // //       printf("\nThread cannot be created : [%s]", strerror(error));
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   i = 0;
// // //   while (i < NUM_THREADS) {
// // //     error = pthread_join(tid[i], &status);
// // //     if (error != 0) {
// // //       printf("ERROR: return code from pthread_join() is %d\n", error);
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
// // //   cout << "Spin lock: Time taken (ms): " << sync_time.load() << "\n";

// // //   // Ticket lock
// // //   var1 = 0;
// // //   var2 = (uint64_t(N) * NUM_THREADS + 1);
// // //   sync_time.store(0);

// // //   lock_obj = new TicketLock();
// // //   i = 0;
// // //   while (i < NUM_THREADS) {
// // //     args[i].m_id = i;
// // //     args[i].m_lock = lock_obj;

// // //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// // //     if (error != 0) {
// // //       printf("\nThread cannot be created : [%s]", strerror(error));
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   i = 0;
// // //   while (i < NUM_THREADS) {
// // //     error = pthread_join(tid[i], &status);
// // //     if (error != 0) {
// // //       printf("ERROR: return code from pthread_join() is %d\n", error);
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
// // //   cout << "Ticket lock: Time taken (ms): " << sync_time.load() << "\n";

// // //   // Array Q lock
// // //   var1 = 0;
// // //   var2 = (uint64_t(N) * NUM_THREADS + 1);
// // //   sync_time.store(0);

// // //   lock_obj = new ArrayQLock();
// // //   i = 0;
// // //   while (i < NUM_THREADS) {
// // //     args[i].m_id = i;
// // //     args[i].m_lock = lock_obj;

// // //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// // //     if (error != 0) {
// // //       printf("\nThread cannot be created : [%s]", strerror(error));
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   i = 0;
// // //   while (i < NUM_THREADS) {
// // //     error = pthread_join(tid[i], &status);
// // //     if (error != 0) {
// // //       printf("ERROR: return code from pthread_join() is %d\n", error);
// // //       exit(EXIT_FAILURE);
// // //     }
// // //     i++;
// // //   }

// // //   cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
// // //   cout << "Array Q lock: Time taken (ms): " << sync_time.load() << "\n";

// // //   pthread_barrier_destroy(&g_barrier);
// // //   pthread_attr_destroy(&attr);

// // //   pthread_exit(NULL);
// // // }



// // #include <cassert>
// // #include <chrono>
// // #include <cstdint>
// // #include <cstdlib>
// // #include <cstring>
// // #include <iostream>
// // #include <pthread.h>
// // #include <atomic>
// // #include <algorithm>

// // using std::cerr;
// // using std::cout;
// // using std::endl;

// // using HR = std::chrono::high_resolution_clock;
// // using HRTimer = HR::time_point;
// // using std::chrono::duration_cast;
// // using std::chrono::microseconds;
// // using std::chrono::milliseconds;

// // #define N (1e7)
// // #define NUM_THREADS (8)

// // // Shared variables
// // uint64_t var1 = 0, var2 = (N * NUM_THREADS + 1);

// // // Abstract base class
// // class LockBase {
// // public:
// //   // Pure virtual function
// //   virtual void acquire(uint16_t tid) = 0;
// //   virtual void release(uint16_t tid) = 0;
// // };

// // typedef struct thr_args {
// //   uint16_t m_id;
// //   LockBase* m_lock;
// // } ThreadArgs;

// // /** Use pthread mutex to implement lock routines */
// // class PthreadMutex : public LockBase {
// // public:
// //   void acquire(uint16_t tid) override { pthread_mutex_lock(&lock); }
// //   void release(uint16_t tid) override { pthread_mutex_unlock(&lock); }

// // private:
// //   pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
// // };

// // /** Filter Lock Implementation */
// // class FilterLock : public LockBase {
// // private:
// //   static const int MAX_THREADS = 128;
// //   std::atomic<int> level[MAX_THREADS];
// //   std::atomic<int> victim[MAX_THREADS];
// //   int n_threads;

// // public:
// //   FilterLock(int num_threads = NUM_THREADS) : n_threads(num_threads) {
// //     for (int i = 0; i < MAX_THREADS; i++) {
// //       level[i].store(0);
// //       victim[i].store(0);
// //     }
// //   }

// //   void acquire(uint16_t tid) override {
// //     for (int i = 1; i < n_threads; i++) {
// //       level[tid].store(i);
// //       victim[i].store(tid);
      
// //       // Spin while conflicts exist
// //       bool wait = true;
// //       while (wait) {
// //         wait = false;
// //         for (int k = 0; k < n_threads; k++) {
// //           if (k != tid && level[k].load() >= i && victim[i].load() == tid) {
// //             wait = true;
// //             break;
// //           }
// //         }
// //       }
// //     }
// //   }

// //   void release(uint16_t tid) override {
// //     level[tid].store(0);
// //   }

// //   ~FilterLock() {}
// // };

// // /** Bakery Lock Implementation */
// // class BakeryLock : public LockBase {
// // private:
// //   static const int MAX_THREADS = 128;
// //   alignas(64) std::atomic<bool> choosing[MAX_THREADS];
// //   alignas(64) std::atomic<int> number[MAX_THREADS];
// //   int n_threads;

// // public:
// //   BakeryLock(int num_threads = NUM_THREADS) : n_threads(num_threads) {
// //     for (int i = 0; i < MAX_THREADS; i++) {
// //       choosing[i].store(false);
// //       number[i].store(0);
// //     }
// //   }

// //   void acquire(uint16_t tid) override {
// //     choosing[tid].store(true);
    
// //     // Find max ticket number
// //     int max_num = 0;
// //     for (int i = 0; i < n_threads; i++) {
// //       int num = number[i].load();
// //       if (num > max_num) {
// //         max_num = num;
// //       }
// //     }
    
// //     number[tid].store(max_num + 1);
// //     choosing[tid].store(false);
    
// //     // Wait for all threads with smaller tickets
// //     for (int i = 0; i < n_threads; i++) {
// //       if (i == tid) continue;
      
// //       // Wait while thread i is choosing
// //       while (choosing[i].load());
      
// //       // Wait while thread i has priority
// //       while (true) {
// //         int num_i = number[i].load();
// //         if (num_i == 0) break;
        
// //         int num_tid = number[tid].load();
// //         if (num_i > num_tid) break;
// //         if (num_i == num_tid && i > tid) break;
// //       }
// //     }
// //   }

// //   void release(uint16_t tid) override {
// //     number[tid].store(0);
// //   }

// //   ~BakeryLock() {}
// // };

// // /** Spin Lock Implementation */
// // class SpinLock : public LockBase {
// // private:
// //   std::atomic_flag lock_flag = ATOMIC_FLAG_INIT;

// // public:
// //   void acquire(uint16_t tid) override {
// //     while (lock_flag.test_and_set(std::memory_order_acquire)) {
// //       // Spin
// //       // Busy-wait until the lock is released
// //     }
// //   }

// //   void release(uint16_t tid) override {
// //     lock_flag.clear(std::memory_order_release);
// //   }

// //   SpinLock() {}
// //   ~SpinLock() {}
// // };

// // /** Ticket Lock Implementation */
// // class TicketLock : public LockBase {
// // private:
// //   alignas(64) std::atomic<uint64_t> next_ticket;
// //   alignas(64) std::atomic<uint64_t> now_serving;

// // public:
// //   TicketLock() : next_ticket(0), now_serving(0) {}

// //   void acquire(uint16_t tid) override {
// //     uint64_t my_ticket = next_ticket.fetch_add(1, std::memory_order_relaxed);
    
// //     while (now_serving.load(std::memory_order_acquire) != my_ticket) {
// //       // Spin with backoff
// //       while (now_serving.load(std::memory_order_relaxed) != my_ticket);
// //     }
// //   }

// //   void release(uint16_t tid) override {
// //     now_serving.fetch_add(1, std::memory_order_release);
// //   }

// //   ~TicketLock() {}
// // };

// // /** Array-based Queue Lock Implementation */
// // class ArrayQLock : public LockBase {
// // private:
// //   static const int MAX_THREADS = 128;
// //   alignas(64) std::atomic<bool> flags[MAX_THREADS];
// //   alignas(64) std::atomic<int> tail;
// //   alignas(64) int my_slot[MAX_THREADS];
// //   int n_threads;

// // public:
// //   ArrayQLock(int num_threads = NUM_THREADS) : n_threads(num_threads), tail(0) {
// //     for (int i = 0; i < MAX_THREADS; i++) {
// //       flags[i].store(false);
// //       my_slot[i] = -1;
// //     }
// //     flags[0].store(true);  // Initially unlocked
// //   }

// //   void acquire(uint16_t tid) override {
// //     int slot = tail.fetch_add(1, std::memory_order_relaxed) % n_threads;
// //     my_slot[tid] = slot;
    
// //     while (!flags[slot].load(std::memory_order_acquire)) {
// //       // Spin
// //     }
    
// //     flags[slot].store(false, std::memory_order_relaxed);
// //   }

// //   void release(uint16_t tid) override {
// //     int slot = my_slot[tid];
// //     int next_slot = (slot + 1) % n_threads;
// //     flags[next_slot].store(true, std::memory_order_release);
// //   }

// //   ~ArrayQLock() {}
// // };

// // /** Estimate the time taken */
// // std::atomic_uint64_t sync_time = 0;

// // inline void critical_section() {
// //   var1++;
// //   var2--;
// // }

// // /** Sync threads at the start to maximize contention */
// // pthread_barrier_t g_barrier;

// // void* thrBody(void* arguments) {
// //   ThreadArgs* tmp = static_cast<ThreadArgs*>(arguments);
// //   if (false) {
// //     cout << "Thread id: " << tmp->m_id << " starting\n";
// //   }

// //   // Wait for all other producer threads to launch before proceeding.
// //   pthread_barrier_wait(&g_barrier);

// //   HRTimer start = HR::now();
// //   for (int i = 0; i < N; i++) {
// //     tmp->m_lock->acquire(tmp->m_id);
// //     critical_section();
// //     tmp->m_lock->release(tmp->m_id);
// //   }
// //   HRTimer end = HR::now();
// //   auto duration = duration_cast<milliseconds>(end - start).count();

// //   // A barrier is not required here
// //   sync_time.fetch_add(duration);
// //   pthread_exit(NULL);
// // }

// // int main() {
// //   int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
// //   if (error != 0) {
// //     cerr << "Error in barrier init.\n";
// //     exit(EXIT_FAILURE);
// //   }

// //   pthread_attr_t attr;
// //   pthread_attr_init(&attr);
// //   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

// //   pthread_t tid[NUM_THREADS];
// //   ThreadArgs args[NUM_THREADS] = {{0}};

// //   // Pthread mutex
// //   LockBase* lock_obj = new PthreadMutex();
// //   uint16_t i = 0;
// //   while (i < NUM_THREADS) {
// //     args[i].m_id = i;
// //     args[i].m_lock = lock_obj;

// //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// //     if (error != 0) {
// //       cerr << "\nThread cannot be created : " << strerror(error) << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   i = 0;
// //   void* status;
// //   while (i < NUM_THREADS) {
// //     error = pthread_join(tid[i], &status);
// //     if (error) {
// //       cerr << "ERROR: return code from pthread_join() is " << error << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   assert(var1 == N * NUM_THREADS && var2 == 1);
// //   // cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
// //   cout << "Pthread mutex: Time taken (ms): " << sync_time << "\n";

// //   // Filter lock
// //   var1 = 0;
// //   var2 = (N * NUM_THREADS + 1);
// //   sync_time.store(0);

// //   lock_obj = new FilterLock(NUM_THREADS);
// //   i = 0;
// //   while (i < NUM_THREADS) {
// //     args[i].m_id = i;
// //     args[i].m_lock = lock_obj;

// //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// //     if (error != 0) {
// //       cerr << "\nThread cannot be created : " << strerror(error) << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   i = 0;
// //   while (i < NUM_THREADS) {
// //     error = pthread_join(tid[i], &status);
// //     if (error) {
// //       cerr << "ERROR: return code from pthread_join() is " << error << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   assert(var1 == N * NUM_THREADS && var2 == 1);
// //   cout << "Filter lock: Time taken (ms): " << sync_time << "\n";

// //   // Bakery lock
// //   var1 = 0;
// //   var2 = (N * NUM_THREADS + 1);
// //   sync_time.store(0);

// //   lock_obj = new BakeryLock(NUM_THREADS);
// //   i = 0;
// //   while (i < NUM_THREADS) {
// //     args[i].m_id = i;
// //     args[i].m_lock = lock_obj;

// //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// //     if (error != 0) {
// //       cerr << "\nThread cannot be created : " << strerror(error) << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   i = 0;
// //   while (i < NUM_THREADS) {
// //     error = pthread_join(tid[i], &status);
// //     if (error) {
// //       cerr << "ERROR: return code from pthread_join() is " << error << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   assert(var1 == N * NUM_THREADS && var2 == 1);
// //   cout << "Bakery lock: Time taken (ms): " << sync_time << "\n";

// //   // Spin lock
// //   var1 = 0;
// //   var2 = (N * NUM_THREADS + 1);
// //   sync_time.store(0);

// //   lock_obj = new SpinLock();
// //   i = 0;
// //   while (i < NUM_THREADS) {
// //     args[i].m_id = i;
// //     args[i].m_lock = lock_obj;

// //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// //     if (error != 0) {
// //       cerr << "\nThread cannot be created : " << strerror(error) << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   i = 0;
// //   while (i < NUM_THREADS) {
// //     error = pthread_join(tid[i], &status);
// //     if (error) {
// //       cerr << "ERROR: return code from pthread_join() is " << error << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   assert(var1 == N * NUM_THREADS && var2 == 1);
// //   cout << "Spin lock: Time taken (ms): " << sync_time << "\n";

// //   // Ticket lock
// //   var1 = 0;
// //   var2 = (N * NUM_THREADS + 1);
// //   sync_time.store(0);

// //   lock_obj = new TicketLock();
// //   i = 0;
// //   while (i < NUM_THREADS) {
// //     args[i].m_id = i;
// //     args[i].m_lock = lock_obj;

// //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// //     if (error != 0) {
// //       cerr << "\nThread cannot be created : " << strerror(error) << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   i = 0;
// //   while (i < NUM_THREADS) {
// //     error = pthread_join(tid[i], &status);
// //     if (error) {
// //       cerr << "ERROR: return code from pthread_join() is " << error << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   assert(var1 == N * NUM_THREADS && var2 == 1);
// //   cout << "Ticket lock: Time taken (ms): " << sync_time << "\n";

// //   // Array Q lock
// //   var1 = 0;
// //   var2 = (N * NUM_THREADS + 1);
// //   sync_time.store(0);

// //   lock_obj = new ArrayQLock(NUM_THREADS);
// //   i = 0;
// //   while (i < NUM_THREADS) {
// //     args[i].m_id = i;
// //     args[i].m_lock = lock_obj;

// //     error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
// //     if (error != 0) {
// //       cerr << "\nThread cannot be created : " << strerror(error) << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   i = 0;
// //   while (i < NUM_THREADS) {
// //     error = pthread_join(tid[i], &status);
// //     if (error) {
// //       cerr << "ERROR: return code from pthread_join() is " << error << "\n";
// //       exit(EXIT_FAILURE);
// //     }
// //     i++;
// //   }

// //   assert(var1 == N * NUM_THREADS && var2 == 1);
// //   cout << "Array Q lock: Time taken (ms): " << sync_time << "\n";

// //   pthread_barrier_destroy(&g_barrier);
// //   pthread_attr_destroy(&attr);

// //   pthread_exit(NULL);
// // }


// rollno-prob3.cpp
// Problem3: implement Filter, Bakery, Spin, Ticket, ArrayQ locks using x86 CAS-like inline asm & mfence
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <vector>
#include <atomic>
#include <immintrin.h>

using std::cerr;
using std::cout;
using std::endl;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

// ---------- config (change ITER and NUM_THREADS as needed) ----------
#define N (1000)          // integer iterations per thread
#define NUM_THREADS (64)

// ---------- shared variables ----------
uint64_t var1 = 0;
uint64_t var2 = (uint64_t(N) * NUM_THREADS + 1);

// ---------- base class ----------
class LockBase {
public:
  virtual void acquire(uint16_t tid) = 0;
  virtual void release(uint16_t tid) = 0;
  virtual ~LockBase() = default;
};

// ---------- helpers: cpu_relax and mfence ----------
static inline void cpu_relax() {
#if defined(__i386__) || defined(__x86_64__)
  __builtin_ia32_pause();
#else
  // fallback
  std::this_thread::yield();
#endif
}

static inline void mfence_fence() {
  __asm__ volatile("mfence" ::: "memory");
}

// ---------- xchg (test-and-set) for 32-bit int ----------
static inline int xchg32(volatile int *ptr, int newv) {
  int oldv;
  __asm__ volatile("xchgl %0, %1"
                   : "=r"(oldv), "+m"(*ptr)
                   : "0"(newv)
                   : "memory");
  return oldv;
}

// ---------- lock xadd for 64-bit (returns previous) ----------
static inline unsigned long xadd64(volatile unsigned long *ptr, unsigned long val) {
  __asm__ volatile("lock xaddq %0, %1"
                   : "+r"(val), "+m"(*ptr)
                   :
                   : "memory");
  return val; // previous value returned in val
}

// ---------- lock cmpxchg for 32-bit: returns 1 on success, 0 on failure ----------
static inline int cas32(volatile int *ptr, int expected, int desired) {
  unsigned char success;
  __asm__ volatile("lock cmpxchgl %2, %1\n\tsete %0"
                   : "=q"(success), "+m"(*ptr), "+a"(expected)
                   : "r"(desired)
                   : "memory");
  return success;
}

// ---------- PthreadMutex wrapper (baseline) ----------
class PthreadMutex : public LockBase {
public:
  PthreadMutex() { pthread_mutex_init(&m, nullptr); }
  ~PthreadMutex() { pthread_mutex_destroy(&m); }
  void acquire(uint16_t) override { pthread_mutex_lock(&m); }
  void release(uint16_t) override { pthread_mutex_unlock(&m); }
private:
  pthread_mutex_t m;
};

class BakeryLock : public LockBase {
public:
  void acquire(uint16_t tid) override {}
  void release(uint16_t tid) override {}

  BakeryLock() {}
  ~BakeryLock() {}
};

class SpinLock : public LockBase {
public:
  void acquire(uint16_t tid) override {}
  void release(uint16_t tid) override {}

  SpinLock() {}
  ~SpinLock() {}
};

class TicketLock : public LockBase {
public:
  void acquire(uint16_t tid) override {}
  void release(uint16_t tid) override {}

  TicketLock() {}
  ~TicketLock() {}
};

class ArrayQLock : public LockBase {
public:
  void acquire(uint16_t tid) override {}
  void release(uint16_t tid) override {}

  ArrayQLock() {}
  ~ArrayQLock() {}
};

// // ---------- SpinLock using xchg ----------
// class SpinLock : public LockBase {
// public:
//   SpinLock() { lock_word = 0; }
//   void acquire(uint16_t) override {
//     while (xchg32(&lock_word, 1) == 1) { cpu_relax(); }
//     // acquired, lock_word==1
//   }
//   void release(uint16_t) override {
//     // release with store (no need for xchg): ensure write ordering
//     __asm__ volatile("" ::: "memory");
//     lock_word = 0;
//   }
// private:
//   volatile int lock_word; // 0 = free, 1 = taken
// };

// // ---------- Ticket lock using lock xadd (fetch-and-add) ----------
// class TicketLock : public LockBase {
// public:
//   TicketLock() { next = 0; serving = 0; }
//   void acquire(uint16_t) override {
//     unsigned long my = xadd64(&next, 1ul); // returns previous
//     // spin on serving
//     while (serving != my) { cpu_relax(); }
//   }
//   void release(uint16_t) override {
//     // increment serving (simple non-atomic store is unsafe; use xadd)
//     xadd64(&serving, 1ul);
//   }
// private:
//   volatile unsigned long next;
//   volatile unsigned long serving;
// };

// // ---------- Padded structures to avoid false sharing ----------
// struct alignas(64) PaddedInt {
//   volatile int v;
//   char pad[64 - sizeof(int)];
// };
// struct alignas(64) PaddedLong {
//   volatile long v;
//   char pad[64 - sizeof(long)];
// };
// struct alignas(64) PaddedChar {
//   volatile char v;
//   char pad[64 - sizeof(char)];
// };

// // ---------- Array-based Queue Lock (Anderson) ----------
// class ArrayQLock : public LockBase {
// public:
//   ArrayQLock(int n = NUM_THREADS) : nslots(n) {
//     flags.resize(nslots);
//     my_slot.resize(nslots);
//     for (int i = 0; i < nslots; ++i) {
//       flags[i].v = 0;
//       my_slot[i] = -1;
//     }
//     flags[0].v = 1; // first slot available
//     tail = 0;
//   }

//   void acquire(uint16_t tid) override {
//     unsigned long pos = xadd64((volatile unsigned long*)&tail, 1ul);
//     int slot = int(pos % nslots);
//     my_slot[tid] = slot;
//     while (!flags[slot].v) { cpu_relax(); }
//     // now have the slot
//   }

//   void release(uint16_t tid) override {
//     int slot = my_slot[tid];
//     flags[slot].v = 0;
//     int nxt = (slot + 1) % nslots;
//     flags[nxt].v = 1;
//   }
// private:
//   int nslots;
//   volatile unsigned long tail;
//   std::vector<PaddedChar> flags;   // each flag on separate cache line
//   std::vector<int> my_slot;        // per-tid slot index
// };

// // ---------- Bakery lock implemented with mfence and plain arrays ----------
// class BakeryLock : public LockBase {
// public:
//   BakeryLock(int n = NUM_THREADS) : nthreads(n) {
//     choosing.resize(nthreads);
//     number.resize(nthreads);
//     for (int i = 0; i < nthreads; ++i) {
//       choosing[i].v = 0;
//       number[i].v = 0;
//     }
//   }

//   void acquire(uint16_t tid) override {
//     choosing[tid].v = 1;
//     mfence_fence(); // prevent reorder of choosing before subsequent read/write

//     // get max
//     int maxv = 0;
//     for (int i = 0; i < nthreads; ++i) {
//       int nv = number[i].v;
//       if (nv > maxv) maxv = nv;
//     }
//     number[tid].v = maxv + 1;
//     mfence_fence(); // make sure number is visible
//     choosing[tid].v = 0;
//     mfence_fence();

//     int mynum = number[tid].v;
//     for (int j = 0; j < nthreads; ++j) {
//       if (j == (int)tid) continue;
//       // wait if j is choosing
//       while (choosing[j].v) cpu_relax();
//       // then wait while j has priority
//       while (true) {
//         int nj = number[j].v;
//         if (nj == 0) break;
//         if (nj < mynum || (nj == mynum && j < (int)tid)) { cpu_relax(); continue; }
//         else break;
//       }
//     }
//   }

//   void release(uint16_t tid) override {
//     number[tid].v = 0;
//     mfence_fence();
//   }

// private:
//   int nthreads;
//   std::vector<PaddedChar> choosing; // using char as boolean
//   std::vector<PaddedInt> number;    // padded per-thread
// };
// ---------- Filter lock using mfence and plain arrays ----------
class FilterLock : public LockBase {
private:
    static const int MAX_THREADS = 64; // adjust if needed
    volatile int level[MAX_THREADS];
    volatile int victim[MAX_THREADS];
    int n_threads;

public:
    FilterLock(int num_threads = 64) : n_threads(num_threads) {
        for (int i = 0; i < MAX_THREADS; ++i) {
            level[i] = 0;
            victim[i] = 0;
        }
    }

    void acquire(uint16_t tid) override {
        for (int L = 1; L < n_threads; ++L) {
            level[tid] = L;
            _mm_mfence();             // prevent store-load reordering
            victim[L] = tid;
            _mm_mfence();

            // Spin while conflict exists
            bool waiting = true;
            while (waiting) {
                waiting = false;
                for (int k = 0; k < n_threads; ++k) {
                    if (k != tid && level[k] >= L && victim[L] == tid) {
                        waiting = true;
                        // Hint to CPU to reduce power / contention
                        asm volatile("pause" ::: "memory");
                        break;
                    }
                }
            }
        }
    }

    void release(uint16_t tid) override {
        level[tid] = 0;
        _mm_mfence(); // ensure all writes are visible before leaving
    }
};

// ---------- harness - mostly unchanged from your base code ----------
std::atomic<uint64_t> sync_time = 0;
pthread_barrier_t g_barrier;

inline void critical_section() {
  // these are simple integer ops inside critical section
  var1++;
  var2--;
}

typedef struct thr_args {
  uint16_t m_id;
  LockBase* m_lock;
} ThreadArgs;

void* thrBody(void* arguments) {
  ThreadArgs* tmp = static_cast<ThreadArgs*>(arguments);
  pthread_barrier_wait(&g_barrier);
  HRTimer start = HR::now();
  for (int i = 0; i < N; ++i) {
    tmp->m_lock->acquire(tmp->m_id);
    critical_section();
    tmp->m_lock->release(tmp->m_id);
  }
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  sync_time.fetch_add((uint64_t)duration);
  pthread_exit(NULL);
  return nullptr;
}

int main() {
  int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
  if (error != 0) {
    cerr << "Error in barrier init.\n";
    exit(EXIT_FAILURE);
  }

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  pthread_t tid[NUM_THREADS];
  ThreadArgs args[NUM_THREADS] = {{0}};

  // Run baseline: Pthread mutex
  LockBase* lock_obj = new PthreadMutex();
  uint16_t i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;
    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      cerr << "\nThread cannot be created : " << strerror(error) << "\n";
      exit(EXIT_FAILURE);
    }
    i++;
  }
  i = 0;
  void* status;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      cerr << "ERROR: return code from pthread_join() is " << error << "\n";
      exit(EXIT_FAILURE);
    }
    i++;
  }
  delete lock_obj;
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  cout << "Pthread mutex: Time taken (ms): " << sync_time.load() << "\n";

  // Filter lock
  var1 = 0;
  var2 = (uint64_t(N) * NUM_THREADS + 1);
  sync_time.store(0);
  lock_obj = new FilterLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;
    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }
  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  cout << "Filter lock: Time taken (ms): " << sync_time.load() << "\n";
  delete lock_obj;

  // Bakery lock
  var1 = 0;
  var2 = (uint64_t(N) * NUM_THREADS + 1);
  sync_time.store(0);
  lock_obj = new BakeryLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;
    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }
  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  cout << "Bakery lock: Time taken (ms): " << sync_time.load() << "\n";
  delete lock_obj;

  // Spin lock
  var1 = 0;
  var2 = (uint64_t(N) * NUM_THREADS + 1);
  sync_time.store(0);
  lock_obj = new SpinLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;
    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }
  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  cout << "Spin lock: Time taken (ms): " << sync_time.load() << "\n";
  delete lock_obj;

  // Ticket lock
  var1 = 0;
  var2 = (uint64_t(N) * NUM_THREADS + 1);
  sync_time.store(0);
  lock_obj = new TicketLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;
    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }
  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  cout << "Ticket lock: Time taken (ms): " << sync_time.load() << "\n";
  delete lock_obj;

  // Array Q lock
  var1 = 0;
  var2 = (uint64_t(N) * NUM_THREADS + 1);
  sync_time.store(0);
  lock_obj = new ArrayQLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;
    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }
  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  cout << "Array Q lock: Time taken (ms): " << sync_time.load() << "\n";
  delete lock_obj;

  pthread_barrier_destroy(&g_barrier);
  pthread_attr_destroy(&attr);

  pthread_exit(NULL);
  return 0;
}
