#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <vector>
#include <immintrin.h> // for _mm_mfence()

using std::cerr;
using std::cout;
using std::endl;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define N (1e4)
#define NUM_THREADS (64)

// Shared variables
uint64_t var1 = 0, var2 = (N * NUM_THREADS + 1);

// Abstract base class
class LockBase {
public:
  // Pure virtual function
  virtual void acquire(uint16_t tid) = 0;
  virtual void release(uint16_t tid) = 0;
};

typedef struct thr_args {
  uint16_t m_id;
  LockBase* m_lock;
} ThreadArgs;

// ========== ATOMIC HELPERS ==========

// Atomic exchange
inline int atomic_xchg(volatile int* addr, int newval) {
    int old;
    asm volatile("lock xchg %0, %1"
                 : "=r"(old), "+m"(*addr)
                 : "0"(newval)
                 : "memory");
    return old;
}

// Atomic compare-and-swap (like cmpxchg)
// Returns true if swapped, false otherwise
inline bool atomic_cmpxchg(volatile int* addr, int expected, int desired) {
    unsigned char result;
    asm volatile("lock cmpxchg %3, %1\n\t"
                 "sete %0"
                 : "=q"(result), "+m"(*addr), "+a"(expected)
                 : "r"(desired)
                 : "memory");
    return result;
}


// fence helper
static inline void cpu_mfence() {
    __asm__ volatile("mfence" ::: "memory");
}

// xadd: atomic fetch-and-increment
static inline int atomic_fetch_and_increment(volatile int *addr) {
    int old = 1;
    __asm__ volatile("lock xaddl %0, %1"
                     : "+r"(old), "+m"(*addr)
                     :
                     : "memory");
    return old;
}

// pause helper
static inline void cpu_pause() {
    __builtin_ia32_pause();
}


/** Use pthread mutex to implement lock routines */
class PthreadMutex : public LockBase {
public:
  void acquire(uint16_t tid) override { pthread_mutex_lock(&lock); }
  void release(uint16_t tid) override { pthread_mutex_unlock(&lock); }

private:
  pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
};

class FilterLock : public LockBase {
private:
    struct PaddedInt { alignas(64) volatile int val; char pad[64 - sizeof(int)]; };
    PaddedInt level[NUM_THREADS];   
    PaddedInt victim[NUM_THREADS];  

public:
    FilterLock() {
        for (int i = 0; i < NUM_THREADS; ++i) {
            level[i].val = 0;
            victim[i].val = -1;
        }
    }

    ~FilterLock() {}

    void acquire(uint16_t tid) override {
        for (int L = 1; L < NUM_THREADS; ++L) {
            level[tid].val = L;
            cpu_mfence();                 // ensure level write visible before victim write
            victim[L].val = tid;
            cpu_mfence();                 // ensure victim write visible

            bool waiting = true;
            while (waiting) {
                waiting = false;
                for (int k = 0; k < NUM_THREADS; ++k) {
                    if (k == tid) continue;
                    // if some other thread is at >= L and we are victim, wait
                    if (level[k].val >= L && victim[L].val == tid) {
                        waiting = true;
                        cpu_pause();
                        break;
                    }
                }
            }
        }
    }

    void release(uint16_t tid) override {
        level[tid].val = 0;
        cpu_mfence();  // make the release visible (store -> load ordering)
    }
};

class BakeryLock : public LockBase {
private:
    struct PaddedBool { alignas(64) volatile bool val; };
    struct PaddedInt  { alignas(64) volatile int  val; };

    PaddedBool choosing[NUM_THREADS];
    PaddedInt number[NUM_THREADS];

    int maxNumber() {
        int mx = 0;
        for (int i = 0; i < NUM_THREADS; i++) {
            int val = number[i].val; // volatile read
            if (val > mx) mx = val;
        }
        return mx;
    }

public:
    BakeryLock() {
        for (int i = 0; i < NUM_THREADS; i++) {
            choosing[i].val = false;
            number[i].val = 0;
        }
    }

    ~BakeryLock() {}

    void acquire(uint16_t tid) override {
        choosing[tid].val = true;
        _mm_mfence(); // ensure store to choosing[tid] is visible before number[]

        number[tid].val = 1 + maxNumber();
        _mm_mfence(); // ensure ticket is visible before clearing choosing[]

        choosing[tid].val = false;
        _mm_mfence(); // ensure clear is visible to others

        for (int j = 0; j < NUM_THREADS; j++) {
            if (j == tid) continue;

            while (choosing[j].val) {
                __builtin_ia32_pause(); // wait until j finishes choosing
            }

            while (number[j].val != 0 &&
                  (number[j].val < number[tid].val ||
                  (number[j].val == number[tid].val && j < tid))) {
                __builtin_ia32_pause();
            }
        }
    }

    void release(uint16_t tid) override {
        number[tid].val = 0;
        _mm_mfence(); // make sure release is globally visible
    }
};


class SpinLock : public LockBase {
private:
    volatile int lock_ = 0;

    inline int xchg(volatile int* addr, int newval) {
        int old;
        __asm__ volatile("xchg %0, %1"
                         : "=r"(old), "+m"(*addr)
                         : "0"(newval)
                         : "memory");
        return old;
    }

public:
    SpinLock() {}
    ~SpinLock() {}

    void acquire(uint16_t tid) override {
        while (true) {
            // try to grab the lock (swap 0 → 1)
            if (xchg(&lock_, 1) == 0) {
                return; 
            }
            // busy-wait until released
            while (lock_) {
                __builtin_ia32_pause(); // polite spin
            }
        }
    }

    void release(uint16_t tid) override {
        lock_ = 0; 
      
    }
};


class TicketLock : public LockBase {
private:
    volatile int next_ticket = 0;
    volatile int now_serving = 0;

    // Atomic fetch-and-add using inline assembly
    inline int fetch_and_add(volatile int* ptr, int inc) {
        int old;
        __asm__ volatile("lock xaddl %0, %1"
                         : "=r"(old), "+m"(*ptr)
                         : "0"(inc)
                         : "memory");
        return old;
    }

public:
    TicketLock() {}
    ~TicketLock() {}

    void acquire(uint16_t tid) override {
        int my_ticket = fetch_and_add(&next_ticket, 1);
        while (now_serving != my_ticket) {
            __builtin_ia32_pause();  // reduce contention
        }
    }

    void release(uint16_t tid) override {
        now_serving++;
      
        
    }
};

class ArrayQLock : public LockBase {
private:
    struct PaddedFlag { alignas(64) volatile int flag; char pad[64 - sizeof(int)]; };
    volatile int tail;                  // ticket dispenser
    PaddedFlag flags[NUM_THREADS];      // flags[i].flag == 1 => that slot holds lock

    // per-thread slot storage
    static thread_local int mySlot;

public:
    ArrayQLock() : tail(0) {
        for (int i = 0; i < NUM_THREADS; ++i) {
            flags[i].flag = 0;
        }
        flags[0].flag = 1; // slot 0 initially allowed
    }

    ~ArrayQLock() {}

    void acquire(uint16_t tid) override {
        int slot = atomic_fetch_and_increment(&tail) % NUM_THREADS;
        mySlot = slot;
        // wait on our slot's flag
        while (flags[slot].flag == 0) {
            cpu_pause();
        }
        // now we have the lock; leave flag[slot] as-is until release
    }

    void release(uint16_t tid) override {
        int slot = mySlot;
        flags[slot].flag = 0;                              // clear own slot
        int next = (slot + 1) % NUM_THREADS;
        cpu_mfence();                                      // ordering: clear before enabling next
        flags[next].flag = 1;                             // enable next slot
        
    }
};

thread_local int ArrayQLock::mySlot = -1;



/** Estimate the time taken */
std::atomic_uint64_t sync_time = 0;

inline void critical_section(uint16_t tid) {
  var1++;
  var2--;
}

/** Sync threads at the start to maximize contention */
pthread_barrier_t g_barrier;

void* thrBody(void* arguments) {
  ThreadArgs* tmp = static_cast<ThreadArgs*>(arguments);
  if (false) {
    cout << "Thread id: " << tmp->m_id << " starting\n";
  }

  // Wait for all other producer threads to launch before proceeding.
  pthread_barrier_wait(&g_barrier);

  HRTimer start = HR::now();
  for (int i = 0; i < N; i++) {
    tmp->m_lock->acquire(tmp->m_id);
    critical_section(tmp->m_id);

    tmp->m_lock->release(tmp->m_id);
    
  }
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();

  // A barrier is not required here
  sync_time.fetch_add(duration);
  pthread_exit(NULL);
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

  // Pthread mutex
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

  assert(var1 == N * NUM_THREADS && var2 == 1);
  // cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  cout << "Pthread mutex: Time taken (us): " << sync_time << "\n";

  // Filter lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
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
  // assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Filter lock: Time taken (us): " << sync_time << "\n";

  // Bakery lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
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
  // assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Bakery lock: Time taken (us): " << sync_time << "\n";

  // Spin lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
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
  // assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Spin lock: Time taken (us): " << sync_time << "\n";

  // Ticket lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
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
  // assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Ticket lock: Time taken (us): " << sync_time << "\n";


  // Array Q lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
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
  // assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Array Q lock: Time taken (us): " << sync_time << "\n";

  pthread_barrier_destroy(&g_barrier);
  pthread_attr_destroy(&attr);

  pthread_exit(NULL);
}
