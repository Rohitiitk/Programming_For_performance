

#include "210874.h"

// Shared data structures
std::queue<std::string> buffer;
std::mutex mtx;
std::condition_variable cv_full, cv_empty;
bool done = false;

// Producer thread
void producer(size_t id, std::ifstream &in, int L) {
    std::vector<std::string> lines;
    std::string line;

    // Read up to L lines (including blank lines)
    for (int i = 0; i < L && std::getline(in, line); i++) {
        lines.push_back(line);
    }

    // Write lines into buffer in chunks of BUFFER_SIZE
    size_t written = 0;
    while (written < lines.size()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_full.wait(lock, [] { return buffer.size() < BUFFER_SIZE; });

        buffer.push(lines[written]);
        written++;

        lock.unlock();
        cv_empty.notify_one();
    }
}

// Consumer thread
void consumer(size_t id, std::ofstream &out) {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);

        cv_empty.wait(lock, [] { return !buffer.empty() || done; });

        if (buffer.empty() && done) break;

        std::string line = buffer.front();
        buffer.pop();

        lock.unlock();
        cv_full.notify_one();

        // Write to file
        out << "Thread " << id << ": " << line << "\n";
    }
}

// Main function
int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input.txt> <output.txt> <L> <C>\n";
        return 1;
    }

    std::ifstream in(argv[1]);
    std::ofstream out(argv[2]);
    int L = std::stoi(argv[3]);  // lines per producer
    int C = std::stoi(argv[4]);  // number of consumers

    if (!in.is_open() || !out.is_open()) {
        std::cerr << "Error opening file.\n";
        return 1;
    }

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    // Launch one producer per L lines until file ends
    while (in.peek() != EOF) {
        producers.emplace_back(producer, producers.size(), std::ref(in), L);
    }

    // Launch consumers
    for (int i = 0; i < C; i++) {
        consumers.emplace_back(consumer, i, std::ref(out));
    }

    // Join producers
    for (auto &t : producers) t.join();

    {
        std::lock_guard<std::mutex> lock(mtx);
        done = true;
    }
    cv_empty.notify_all();

    // Join consumers
    for (auto &t : consumers) t.join();

    return 0;
}
