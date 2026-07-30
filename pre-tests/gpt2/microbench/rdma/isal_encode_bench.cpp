#include <isa-l/erasure_code.h>

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    int threads = 128;
    int k = 2;
    int m = 2;
    std::size_t block_size = 2ULL * 1024 * 1024;
    int warmup = 3;
    int iters = 10;
};

struct ThreadContext {
    int thread_id;
    int cpu;
    const Options* options;
    pthread_barrier_t* start_barrier;
    pthread_barrier_t* finish_barrier;
    std::uint64_t checksum = 0;
    std::string error;
};

std::size_t parse_size(const std::string& value) {
    if (value.empty()) throw std::invalid_argument("empty size");
    std::string number = value;
    std::size_t multiplier = 1;
    const char suffix = static_cast<char>(std::tolower(number.back()));
    if (suffix == 'k' || suffix == 'm' || suffix == 'g') {
        number.pop_back();
        multiplier = suffix == 'k' ? 1024ULL : suffix == 'm' ? 1024ULL * 1024 : 1024ULL * 1024 * 1024;
    }
    const double parsed = std::stod(number);
    if (parsed <= 0 || parsed > static_cast<double>(std::numeric_limits<std::size_t>::max() / multiplier)) {
        throw std::invalid_argument("invalid size: " + value);
    }
    return static_cast<std::size_t>(parsed * multiplier);
}

int parse_positive(const char* name, const char* value) {
    const long parsed = std::stol(value);
    if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<int>(parsed);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--threads") options.threads = parse_positive("threads", next());
        else if (arg == "--k") options.k = parse_positive("k", next());
        else if (arg == "--m") options.m = parse_positive("m", next());
        else if (arg == "--block") options.block_size = parse_size(next());
        else if (arg == "--warmup") options.warmup = parse_positive("warmup", next());
        else if (arg == "--iters") options.iters = parse_positive("iters", next());
        else if (arg == "--help") {
            std::cout << "Usage: isal_encode_bench [--threads N] [--k N] [--m N] [--block SIZE] [--warmup N] [--iters N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.k + options.m > 255) throw std::invalid_argument("k + m must not exceed 255");
    return options;
}

std::vector<int> allowed_cpus() {
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
        throw std::runtime_error(std::string("sched_getaffinity failed: ") + std::strerror(errno));
    }
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &affinity)) cpus.push_back(cpu);
    }
    return cpus;
}

void* worker(void* opaque) {
    auto* context = static_cast<ThreadContext*>(opaque);
    try {
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(context->cpu, &affinity);
        const int affinity_status = pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
        if (affinity_status != 0) throw std::runtime_error(std::string("pthread_setaffinity_np failed: ") + std::strerror(affinity_status));

        const Options& options = *context->options;
        std::vector<std::vector<unsigned char>> inputs(options.k, std::vector<unsigned char>(options.block_size));
        std::vector<std::vector<unsigned char>> outputs(options.m, std::vector<unsigned char>(options.block_size));
        std::vector<unsigned char*> data(options.k);
        std::vector<unsigned char*> coding(options.m);
        for (int i = 0; i < options.k; ++i) {
            std::fill(inputs[i].begin(), inputs[i].end(), static_cast<unsigned char>((context->thread_id * 17 + i * 29 + 1) & 0xff));
            data[i] = inputs[i].data();
        }
        for (int i = 0; i < options.m; ++i) coding[i] = outputs[i].data();

        std::vector<unsigned char> matrix(static_cast<std::size_t>(options.k + options.m) * options.k);
        gf_gen_rs_matrix(matrix.data(), options.k + options.m, options.k);
        std::vector<unsigned char> tables(32ULL * options.k * options.m);
        ec_init_tables(options.k, options.m, matrix.data() + options.k * options.k, tables.data());

        for (int i = 0; i < options.warmup; ++i) {
            ec_encode_data(static_cast<int>(options.block_size), options.k, options.m, tables.data(), data.data(), coding.data());
        }
        pthread_barrier_wait(context->start_barrier);
        for (int i = 0; i < options.iters; ++i) {
            ec_encode_data(static_cast<int>(options.block_size), options.k, options.m, tables.data(), data.data(), coding.data());
        }
        pthread_barrier_wait(context->finish_barrier);

        std::uint64_t checksum = 1469598103934665603ULL;
        for (const auto& output : outputs) {
            for (unsigned char byte : output) checksum = (checksum ^ byte) * 1099511628211ULL;
        }
        context->checksum = checksum;
    } catch (const std::exception& error) {
        context->error = error.what();
        pthread_barrier_wait(context->start_barrier);
        pthread_barrier_wait(context->finish_barrier);
    }
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        if (options.block_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("block size exceeds ISA-L int length limit");
        }
        const std::vector<int> cpus = allowed_cpus();
        if (options.threads > static_cast<int>(cpus.size())) {
            throw std::runtime_error("requested threads=" + std::to_string(options.threads) +
                                     " exceeds allowed cpuset CPUs=" + std::to_string(cpus.size()));
        }

        pthread_barrier_t start_barrier;
        pthread_barrier_t finish_barrier;
        pthread_barrier_init(&start_barrier, nullptr, options.threads + 1);
        pthread_barrier_init(&finish_barrier, nullptr, options.threads + 1);
        std::vector<pthread_t> threads(options.threads);
        std::vector<ThreadContext> contexts;
        contexts.reserve(options.threads);
        for (int i = 0; i < options.threads; ++i) {
            contexts.push_back({i, cpus[i], &options, &start_barrier, &finish_barrier, 0, {}});
            const int status = pthread_create(&threads[i], nullptr, worker, &contexts.back());
            if (status != 0) throw std::runtime_error(std::string("pthread_create failed: ") + std::strerror(status));
        }

        pthread_barrier_wait(&start_barrier);
        const auto start = std::chrono::steady_clock::now();
        pthread_barrier_wait(&finish_barrier);
        const auto finish = std::chrono::steady_clock::now();
        for (pthread_t thread : threads) pthread_join(thread, nullptr);
        pthread_barrier_destroy(&start_barrier);
        pthread_barrier_destroy(&finish_barrier);
        for (const auto& context : contexts) {
            if (!context.error.empty()) throw std::runtime_error("worker " + std::to_string(context.thread_id) + ": " + context.error);
        }

        const double seconds = std::chrono::duration<double>(finish - start).count();
        const double gib = 1024.0 * 1024.0 * 1024.0;
        const double source_bytes = static_cast<double>(options.threads) * options.k * options.block_size * options.iters;
        const double parity_bytes = static_cast<double>(options.threads) * options.m * options.block_size * options.iters;
        const std::uint64_t checksum = std::accumulate(contexts.begin(), contexts.end(), std::uint64_t{0},
            [](std::uint64_t sum, const ThreadContext& context) { return sum ^ context.checksum; });
        std::cout.setf(std::ios::fixed);
        std::cout.precision(6);
        std::cout << "RESULT,mode=isal_encode,threads=" << options.threads
                  << ",allowed_cpus=" << cpus.size() << ",k=" << options.k << ",m=" << options.m
                  << ",block_bytes=" << options.block_size << ",warmup=" << options.warmup
                  << ",iters=" << options.iters << ",wall_s=" << seconds
                  << ",source_gib_s=" << source_bytes / gib / seconds
                  << ",parity_gib_s=" << parity_bytes / gib / seconds
                  << ",memory_gib_s=" << (source_bytes + parity_bytes) / gib / seconds
                  << ",per_thread_source_gib_s=" << source_bytes / gib / seconds / options.threads
                  << ",checksum=" << checksum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR," << error.what() << '\n';
        return 1;
    }
}
