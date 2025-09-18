
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

// Shared buffer size (M)
constexpr size_t BUFFER_SIZE = 5;

// Function declarations
void producer(size_t id, std::ifstream &in, int L);
void consumer(size_t id, std::ofstream &out);
int main(int argc, char* argv[]);

