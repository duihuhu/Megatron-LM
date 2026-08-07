#include <isa-l/erasure_code.h>

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
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
#include <thread>
#include <vector>

namespace {

struct Options {
    int workers = 16;
    int k = 2;
    int m = 2;
    std::size_t total_bytes = 2774274048ULL;
    int jobs = 102;
    int warmup = 2;
    int repeats = 5;
    std::vector<int> cpus;
    int init_cpu = -1;
    std::string overlap = "none";
    std::vector<int> overlap_cpus;
    std::size_t overlap_bytes = 64ULL * 1024 * 1024;
};

struct Job {
    int len = 0;
    std::vector<std::vector<unsigned char>> inputs;
    std::vector<std::vector<unsigned char>> outputs;
    std::vector<unsigned char*> data;
    std::vector<unsigned char*> coding;
};

struct Span {
    int job = 0;
    int offset = 0;
    int length = 0;
};

struct WorkerContext {
    int id = 0;
    int cpu = 0;
    const Options* options = nullptr;
    std::vector<unsigned char>* tables = nullptr;
    std::vector<Job>* jobs = nullptr;
    const std::vector<Span>* spans = nullptr;
    pthread_barrier_t* barrier = nullptr;
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

std::vector<int> parse_cpu_list(const std::string& value) {
    std::vector<int> cpus;
    std::size_t begin = 0;
    while (begin < value.size()) {
        const std::size_t comma = value.find(',', begin);
        const std::string token = value.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        const std::size_t dash = token.find('-');
        const int first = std::stoi(token.substr(0, dash));
        const int last = dash == std::string::npos ? first : std::stoi(token.substr(dash + 1));
        if (first < 0 || last < first || last >= CPU_SETSIZE) throw std::invalid_argument("invalid CPU list: " + value);
        for (int cpu = first; cpu <= last; ++cpu) cpus.push_back(cpu);
        begin = comma == std::string::npos ? value.size() : comma + 1;
    }
    if (cpus.empty()) throw std::invalid_argument("CPU list must not be empty");
    return cpus;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--workers" || arg == "--threads") options.workers = parse_positive("workers", next());
        else if (arg == "--k") options.k = parse_positive("k", next());
        else if (arg == "--m") options.m = parse_positive("m", next());
        else if (arg == "--total") options.total_bytes = parse_size(next());
        else if (arg == "--jobs") options.jobs = parse_positive("jobs", next());
        else if (arg == "--warmup") options.warmup = parse_positive("warmup", next());
        else if (arg == "--repeats" || arg == "--iters") options.repeats = parse_positive("repeats", next());
        else if (arg == "--cpus") options.cpus = parse_cpu_list(next());
        else if (arg == "--init-cpu") options.init_cpu = std::stoi(next());
        else if (arg == "--overlap") options.overlap = next();
        else if (arg == "--overlap-cpus") options.overlap_cpus = parse_cpu_list(next());
        else if (arg == "--overlap-bytes") options.overlap_bytes = parse_size(next());
        else if (arg == "--help") {
            std::cout << "Usage: isal_encode_bench [--workers N] [--cpus LIST] [--k N] [--m N] "
                         "[--total SIZE] [--jobs N] [--warmup N] [--repeats N] [--init-cpu CPU] "
                         "[--overlap none|poll|memory] [--overlap-cpus LIST] [--overlap-bytes SIZE]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.k + options.m > 255) throw std::invalid_argument("k + m must not exceed 255");
    if (options.total_bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()) * options.jobs) {
        throw std::invalid_argument("job lengths may exceed ISA-L int limit");
    }
    if (options.overlap != "none" && options.overlap != "poll" && options.overlap != "memory") {
        throw std::invalid_argument("overlap must be none, poll, or memory");
    }
    return options;
}

std::vector<int> allowed_cpus() {
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
        throw std::runtime_error(std::string("sched_getaffinity failed: ") + std::strerror(errno));
    }
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &affinity)) cpus.push_back(cpu);
    return cpus;
}

void validate_cpus(const std::vector<int>& requested, const std::vector<int>& allowed, const char* name) {
    for (int cpu : requested) {
        if (std::find(allowed.begin(), allowed.end(), cpu) == allowed.end()) {
            throw std::runtime_error(std::string(name) + " CPU " + std::to_string(cpu) + " is outside allowed cpuset");
        }
    }
}

void pin_current_thread(int cpu) {
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpu, &affinity);
    const int status = pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
    if (status != 0) throw std::runtime_error(std::string("pthread_setaffinity_np failed: ") + std::strerror(status));
}

std::vector<std::vector<Span>> build_plan(const std::vector<Job>& jobs, int workers, std::size_t total_bytes) {
    std::vector<std::vector<Span>> plan(workers);
    const std::uint64_t base = total_bytes / workers;
    const std::uint64_t rem = total_bytes % workers;
    for (int worker = 0; worker < workers; ++worker) {
        const std::uint64_t worker_begin = static_cast<std::uint64_t>(worker) * base + std::min<std::uint64_t>(worker, rem);
        const std::uint64_t worker_end = worker_begin + base + (static_cast<std::uint64_t>(worker) < rem ? 1 : 0);
        std::uint64_t job_begin = 0;
        for (std::size_t job = 0; job < jobs.size() && job_begin < worker_end; ++job) {
            const std::uint64_t job_end = job_begin + jobs[job].len;
            const std::uint64_t begin = std::max(worker_begin, job_begin);
            const std::uint64_t end = std::min(worker_end, job_end);
            if (begin < end) plan[worker].push_back({static_cast<int>(job), static_cast<int>(begin - job_begin), static_cast<int>(end - begin)});
            job_begin = job_end;
        }
    }
    return plan;
}

void* encode_worker(void* opaque) {
    auto* context = static_cast<WorkerContext*>(opaque);
    try {
        pin_current_thread(context->cpu);
        for (const Span& span : *context->spans) {
            Job& job = (*context->jobs)[span.job];
            for (int i = 0; i < context->options->k; ++i) {
                std::memset(job.inputs[i].data() + span.offset, (context->id * 17 + i * 29 + 1) & 0xff, span.length);
            }
            for (int i = 0; i < context->options->m; ++i) {
                std::memset(job.outputs[i].data() + span.offset, 0, span.length);
            }
        }
        pthread_barrier_wait(context->barrier);
        const int rounds = context->options->warmup + context->options->repeats;
        for (int round = 0; round < rounds; ++round) {
            pthread_barrier_wait(context->barrier);
            for (const Span& span : *context->spans) {
                Job& job = (*context->jobs)[span.job];
                std::vector<unsigned char*> data(context->options->k);
                std::vector<unsigned char*> coding(context->options->m);
                for (int i = 0; i < context->options->k; ++i) data[i] = job.data[i] + span.offset;
                for (int i = 0; i < context->options->m; ++i) coding[i] = job.coding[i] + span.offset;
                ec_encode_data(span.length, context->options->k, context->options->m,
                               context->tables->data(), data.data(), coding.data());
            }
            pthread_barrier_wait(context->barrier);
        }
    } catch (const std::exception& error) {
        context->error = error.what();
        pthread_barrier_wait(context->barrier);
        const int rounds = context->options->warmup + context->options->repeats;
        for (int round = 0; round < rounds; ++round) {
            pthread_barrier_wait(context->barrier);
            pthread_barrier_wait(context->barrier);
        }
    }
    return nullptr;
}

std::string join_cpus(const std::vector<int>& cpus) {
    std::string result;
    for (std::size_t i = 0; i < cpus.size(); ++i) {
        if (i) result += ':';
        result += std::to_string(cpus[i]);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_args(argc, argv);
        const std::vector<int> allowed = allowed_cpus();
        if (options.cpus.empty()) options.cpus.assign(allowed.begin(), allowed.begin() + std::min<int>(options.workers, allowed.size()));
        if (static_cast<int>(options.cpus.size()) != options.workers) throw std::invalid_argument("CPU list length must equal workers");
        if (options.overlap != "none" && options.overlap_cpus.empty()) throw std::invalid_argument("overlap CPU list is required");
        validate_cpus(options.cpus, allowed, "worker");
        validate_cpus(options.overlap_cpus, allowed, "overlap");
        if (options.init_cpu >= 0) {
            validate_cpus(std::vector<int>{options.init_cpu}, allowed, "init");
            pin_current_thread(options.init_cpu);
        }

        std::vector<unsigned char> matrix(static_cast<std::size_t>(options.k + options.m) * options.k);
        gf_gen_rs_matrix(matrix.data(), options.k + options.m, options.k);
        std::vector<unsigned char> tables(32ULL * options.k * options.m);
        ec_init_tables(options.k, options.m, matrix.data() + options.k * options.k, tables.data());

        std::vector<Job> jobs(options.jobs);
        const std::size_t base = options.total_bytes / options.jobs;
        const std::size_t rem = options.total_bytes % options.jobs;
        for (int j = 0; j < options.jobs; ++j) {
            const std::size_t len = base + (static_cast<std::size_t>(j) < rem ? 1 : 0);
            if (len > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("job exceeds ISA-L int length limit");
            jobs[j].len = static_cast<int>(len);
            jobs[j].inputs.resize(options.k);
            jobs[j].outputs.resize(options.m);
            jobs[j].data.resize(options.k);
            jobs[j].coding.resize(options.m);
            for (int i = 0; i < options.k; ++i) {
                jobs[j].inputs[i].resize(len);
                jobs[j].data[i] = jobs[j].inputs[i].data();
            }
            for (int i = 0; i < options.m; ++i) {
                jobs[j].outputs[i].resize(len);
                jobs[j].coding[i] = jobs[j].outputs[i].data();
            }
        }
        const auto plan = build_plan(jobs, options.workers, options.total_bytes);

        std::atomic<bool> overlap_stop{false};
        std::atomic<std::uint64_t> overlap_ops{0};
        std::vector<std::thread> overlap_threads;
        for (int cpu : options.overlap_cpus) {
            overlap_threads.emplace_back([&, cpu]() {
                pin_current_thread(cpu);
                if (options.overlap == "memory") {
                    std::vector<unsigned char> buffer(options.overlap_bytes, 1);
                    std::size_t offset = 0;
                    while (!overlap_stop.load(std::memory_order_relaxed)) {
                        buffer[offset] = static_cast<unsigned char>(buffer[offset] + 1);
                        offset = (offset + 64) % buffer.size();
                        overlap_ops.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    while (!overlap_stop.load(std::memory_order_relaxed)) {
                        overlap_ops.fetch_add(1, std::memory_order_relaxed);
                        asm volatile("pause" ::: "memory");
                    }
                }
            });
        }

        pthread_barrier_t barrier;
        pthread_barrier_init(&barrier, nullptr, options.workers + 1);
        std::vector<pthread_t> threads(options.workers);
        std::vector<WorkerContext> contexts(options.workers);
        for (int i = 0; i < options.workers; ++i) {
            contexts[i] = {i, options.cpus[i], &options, &tables, &jobs, &plan[i], &barrier, {}};
            const int status = pthread_create(&threads[i], nullptr, encode_worker, &contexts[i]);
            if (status != 0) throw std::runtime_error(std::string("pthread_create failed: ") + std::strerror(status));
        }
        pthread_barrier_wait(&barrier);

        std::vector<double> wall_seconds;
        for (int round = 0; round < options.warmup + options.repeats; ++round) {
            pthread_barrier_wait(&barrier);
            const auto start = std::chrono::steady_clock::now();
            pthread_barrier_wait(&barrier);
            const auto finish = std::chrono::steady_clock::now();
            if (round >= options.warmup) wall_seconds.push_back(std::chrono::duration<double>(finish - start).count());
        }
        for (pthread_t thread : threads) pthread_join(thread, nullptr);
        pthread_barrier_destroy(&barrier);
        overlap_stop.store(true, std::memory_order_relaxed);
        for (auto& thread : overlap_threads) thread.join();
        for (const auto& context : contexts) if (!context.error.empty()) throw std::runtime_error("worker " + std::to_string(context.id) + ": " + context.error);

        std::uint64_t checksum = 0;
        for (const Job& job : jobs) {
            for (const auto& output : job.outputs) checksum = checksum * 1315423911ULL + output.front() + output.back();
        }
        const double total_s = std::accumulate(wall_seconds.begin(), wall_seconds.end(), 0.0);
        const double mean_s = total_s / wall_seconds.size();
        const double min_s = *std::min_element(wall_seconds.begin(), wall_seconds.end());
        const double max_s = *std::max_element(wall_seconds.begin(), wall_seconds.end());
        const double gib = 1024.0 * 1024.0 * 1024.0;
        const double source_bytes = static_cast<double>(options.total_bytes) * options.k;
        const double parity_bytes = static_cast<double>(options.total_bytes) * options.m;
        std::cout.setf(std::ios::fixed);
        std::cout.precision(6);
        std::cout << "RESULT,mode=frcheck_batch,workers=" << options.workers
                  << ",cpus=" << join_cpus(options.cpus)
                  << ",k=" << options.k << ",m=" << options.m << ",jobs=" << options.jobs
                  << ",logical_range_bytes=" << options.total_bytes
                  << ",init_cpu=" << options.init_cpu
                  << ",warmup=" << options.warmup << ",repeats=" << options.repeats
                  << ",wall_mean_s=" << mean_s << ",wall_min_s=" << min_s << ",wall_max_s=" << max_s
                  << ",range_gib_s=" << options.total_bytes / gib / mean_s
                  << ",source_gib_s=" << source_bytes / gib / mean_s
                  << ",parity_gib_s=" << parity_bytes / gib / mean_s
                  << ",memory_gib_s=" << (source_bytes + parity_bytes) / gib / mean_s
                  << ",overlap=" << options.overlap << ",overlap_cpus=" << join_cpus(options.overlap_cpus)
                  << ",overlap_ops=" << overlap_ops.load(std::memory_order_relaxed)
                  << ",checksum=" << checksum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR," << error.what() << '\n';
        return 1;
    }
}
