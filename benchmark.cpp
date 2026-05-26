// =============================================================================
// benchmark.cpp
// VecDB Phase 3 Benchmark
//
// Usage:  ./benchmark <port>        (server must already be running)
//         ./benchmark <port> <seed>  (optional custom random seed)
//
// What it does:
//   1. Generates 50,000 random D=64 vectors (normal distribution, seed=42)
//   2. Inserts them via ADD commands (measures insertion throughput)
//   3. Sends BUILD to build the IVF index (measures build time)
//   4. Generates 100 random query vectors
//   5. For each query, runs:
//        - BRUTE top-10
//        - IVF top-10  nprobe = 1, 5, 10, 25
//   6. Computes and prints:
//        - avg query latency (ms)
//        - recall@10 vs BRUTE
//        - avg vectors scanned
//
// Recall@10:  |IVF_ids ∩ BRUTE_ids| / 10   averaged over all queries
// =============================================================================

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <numeric>

// POSIX networking
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Benchmark parameters
// ---------------------------------------------------------------------------
static constexpr int   BM_N        = 50000;  // number of vectors to insert
static constexpr int   BM_D        = 64;     // dimension
static constexpr int   BM_QUERIES  = 100;    // query count
static constexpr int   BM_K        = 10;     // top-k
static const int       NPROBES[]   = {1, 5, 10, 25}; // IVF nprobe settings

// ---------------------------------------------------------------------------
// TCP helpers — minimal blocking client (same pattern as client.cpp)
// ---------------------------------------------------------------------------

static int connect_to_server(const char* host, int port) {
    hostent* he = gethostbyname(host);
    if (!he) { std::cerr << "Cannot resolve " << host << "\n"; return -1; }

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);
    std::memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);

    if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect"); close(fd); return -1;
    }
    return fd;
}

// Read exactly one '\n'-terminated line from fd
static std::string recv_line(int fd) {
    std::string line;
    char c;
    while (true) {
        ssize_t n = recv(fd, &c, 1, 0);
        if (n <= 0) return "";
        if (c == '\n') break;
        if (c != '\r') line += c;
    }
    return line;
}

// Send a string + '\n'
static void send_line(int fd, const std::string& s) {
    std::string out = s + "\n";
    send(fd, out.c_str(), out.size(), MSG_NOSIGNAL);
}

// Send a command and receive the full response.
// Returns the status line (first line).
// If n_data_rows > 0, reads that many additional lines into `rows`.
static std::string transact(int fd, const std::string& cmd,
                             std::vector<std::string>* rows = nullptr) {
    send_line(fd, cmd);
    std::string status = recv_line(fd);

    // Figure out how many data rows follow from the status line
    // "OK <N> results..." → read N rows
    int n_rows = 0;
    if (status.rfind("OK ", 0) == 0) {
        const char* p = status.c_str() + 3;
        char* end;
        long n = std::strtol(p, &end, 10);
        if (end != p && std::strncmp(end, " results", 8) == 0)
            n_rows = (int)n;
    }
    for (int i = 0; i < n_rows; ++i) {
        std::string row = recv_line(fd);
        if (rows) rows->push_back(row);
    }
    return status;
}

// Parse "id=X" from a result row. Returns 0 on parse failure.
static uint64_t parse_id_from_row(const std::string& row) {
    auto pos = row.find("id=");
    if (pos == std::string::npos) return 0;
    return std::stoull(row.substr(pos + 3));
}

// Parse "scanned=X" from a status line. Returns 0 if not present.
static int parse_scanned(const std::string& status) {
    auto pos = status.find("scanned=");
    if (pos == std::string::npos) return 0;
    return std::stoi(status.substr(pos + 8));
}

// ---------------------------------------------------------------------------
// Generate N random D-dimensional float vectors (standard normal)
// ---------------------------------------------------------------------------
static std::vector<std::vector<float>> generate_vectors(int N, int D, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::vector<float>> vecs(N, std::vector<float>(D));
    for (auto& v : vecs)
        for (float& x : v) x = dist(rng);
    return vecs;
}

// ---------------------------------------------------------------------------
// Format a float vector as space-separated string
// ---------------------------------------------------------------------------
static std::string vec_to_str(const std::vector<float>& v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ' ';
        oss << v[i];
    }
    return oss.str();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <port> [seed]\n";
        return 1;
    }
    int      port = std::atoi(argv[1]);
    uint64_t seed = (argc == 3) ? (uint64_t)std::stoull(argv[2]) : 42ULL;

    std::cout << "==========================================================\n";
    std::cout << " VecDB Phase 3 Benchmark\n";
    std::cout << " N=" << BM_N << "  D=" << BM_D
              << "  queries=" << BM_QUERIES
              << "  k=" << BM_K << "  seed=" << seed << "\n";
    std::cout << "==========================================================\n\n";

    // -----------------------------------------------------------------------
    // Step 1: Connect to server
    // -----------------------------------------------------------------------
    int fd = connect_to_server("127.0.0.1", port);
    if (fd < 0) return 1;

    // Consume welcome banner
    std::string banner = recv_line(fd);
    std::cout << "[bench] Connected: " << banner << "\n";

    // -----------------------------------------------------------------------
    // Step 2: Generate dataset
    // -----------------------------------------------------------------------
    std::cout << "[bench] Generating " << BM_N << " random vectors (D=" << BM_D << ")...\n";
    auto dataset = generate_vectors(BM_N, BM_D, seed);
    auto queries = generate_vectors(BM_QUERIES, BM_D, seed + 1);

    // -----------------------------------------------------------------------
    // Step 3: Insert vectors — measure throughput
    // -----------------------------------------------------------------------
    std::cout << "[bench] Inserting " << BM_N << " vectors...\n";
    auto t_ins_start = std::chrono::high_resolution_clock::now();

    int insert_errors = 0;
    for (int i = 0; i < BM_N; ++i) {
        std::string cmd = "ADD " + std::to_string(i) + " " + vec_to_str(dataset[i]);
        std::string resp = transact(fd, cmd);
        if (resp.rfind("OK", 0) != 0) insert_errors++;

        // Progress dot every 5000 inserts
        if ((i + 1) % 5000 == 0) {
            std::cout << "  inserted " << (i + 1) << "/" << BM_N << "\n";
            std::cout.flush();
        }
    }

    auto t_ins_end = std::chrono::high_resolution_clock::now();
    double ins_ms = std::chrono::duration<double, std::milli>(t_ins_end - t_ins_start).count();
    double ins_per_sec = (BM_N / ins_ms) * 1000.0;

    std::cout << "\n[bench] Insertion: " << (int)ins_ms << " ms"
              << "  throughput=" << (int)ins_per_sec << " vectors/sec"
              << "  errors=" << insert_errors << "\n\n";

    // -----------------------------------------------------------------------
    // Step 4: BUILD IVF index
    // -----------------------------------------------------------------------
    std::cout << "[bench] Running BUILD...\n";
    auto t_build_start = std::chrono::high_resolution_clock::now();
    std::string build_resp = transact(fd, "BUILD");
    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count();
    std::cout << "[bench] BUILD response: " << build_resp << "\n";
    std::cout << "[bench] BUILD wall time: " << (int)build_ms << " ms\n\n";

    if (build_resp.rfind("OK", 0) != 0) {
        std::cerr << "[bench] BUILD failed — aborting\n";
        close(fd);
        return 1;
    }

    // -----------------------------------------------------------------------
    // Step 5: Run BRUTE queries and collect ground truth
    // -----------------------------------------------------------------------
    std::cout << "[bench] Running " << BM_QUERIES << " BRUTE queries for ground truth...\n";

    // ground_truth[q] = set of IDs returned by BRUTE for query q
    std::vector<std::set<uint64_t>> ground_truth(BM_QUERIES);
    std::vector<double> brute_times(BM_QUERIES);

    for (int q = 0; q < BM_QUERIES; ++q) {
        std::string cmd = "SEARCH " + vec_to_str(queries[q])
                        + " " + std::to_string(BM_K) + " BRUTE";
        std::vector<std::string> rows;

        auto t0 = std::chrono::high_resolution_clock::now();
        transact(fd, cmd, &rows);
        auto t1 = std::chrono::high_resolution_clock::now();
        brute_times[q] = std::chrono::duration<double, std::milli>(t1 - t0).count();

        for (auto& row : rows)
            ground_truth[q].insert(parse_id_from_row(row));
    }

    double avg_brute_ms = std::accumulate(brute_times.begin(), brute_times.end(), 0.0)
                          / BM_QUERIES;
    std::cout << "[bench] BRUTE avg latency: " << std::fixed << std::setprecision(3)
              << avg_brute_ms << " ms\n\n";

    // -----------------------------------------------------------------------
    // Step 6: Run IVF queries for each nprobe setting
    // -----------------------------------------------------------------------
    struct BenchRow {
        int    nprobe;
        double avg_ms;
        double recall;
        double avg_scanned;
    };
    std::vector<BenchRow> results_table;

    for (int nprobe : NPROBES) {
        std::cout << "[bench] IVF nprobe=" << nprobe << " ...\n";

        std::vector<double> times(BM_QUERIES);
        std::vector<double> recalls(BM_QUERIES);
        std::vector<double> scanned_counts(BM_QUERIES);

        for (int q = 0; q < BM_QUERIES; ++q) {
            std::string cmd = "SEARCH " + vec_to_str(queries[q])
                            + " " + std::to_string(BM_K)
                            + " IVF " + std::to_string(nprobe);
            std::vector<std::string> rows;

            auto t0 = std::chrono::high_resolution_clock::now();
            std::string status = transact(fd, cmd, &rows);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[q] = std::chrono::duration<double, std::milli>(t1 - t0).count();

            scanned_counts[q] = parse_scanned(status);

            // Recall@k: fraction of BRUTE results found by IVF
            std::set<uint64_t> ivf_ids;
            for (auto& row : rows) ivf_ids.insert(parse_id_from_row(row));

            int overlap = 0;
            for (uint64_t id : ivf_ids)
                if (ground_truth[q].count(id)) overlap++;

            int gt_size = (int)ground_truth[q].size();
            recalls[q] = gt_size > 0 ? (double)overlap / gt_size : 1.0;
        }

        BenchRow row;
        row.nprobe       = nprobe;
        row.avg_ms       = std::accumulate(times.begin(), times.end(), 0.0) / BM_QUERIES;
        row.recall       = std::accumulate(recalls.begin(), recalls.end(), 0.0) / BM_QUERIES;
        row.avg_scanned  = std::accumulate(scanned_counts.begin(), scanned_counts.end(), 0.0)
                           / BM_QUERIES;
        results_table.push_back(row);
    }

    // -----------------------------------------------------------------------
    // Step 7: Print results table
    // -----------------------------------------------------------------------
    std::cout << "\n";
    std::cout << "==========================================================\n";
    std::cout << " BENCHMARK RESULTS\n";
    std::cout << "==========================================================\n";
    std::cout << std::left
              << std::setw(8)  << "MODE"
              << std::setw(10) << "NPROBE"
              << std::setw(12) << "AVG_MS"
              << std::setw(14) << "RECALL@10"
              << std::setw(14) << "AVG_SCANNED"
              << "\n";
    std::cout << std::string(58, '-') << "\n";

    // BRUTE row
    {
        std::ostringstream ms_s, recall_s, scan_s;
        ms_s    << std::fixed << std::setprecision(3) << avg_brute_ms;
        recall_s << "1.000";
        scan_s  << BM_N;
        std::cout << std::left
                  << std::setw(8)  << "BRUTE"
                  << std::setw(10) << "-"
                  << std::setw(12) << ms_s.str()
                  << std::setw(14) << recall_s.str()
                  << std::setw(14) << scan_s.str()
                  << "\n";
    }

    for (const auto& r : results_table) {
        std::ostringstream ms_s, recall_s, scan_s;
        ms_s    << std::fixed << std::setprecision(3) << r.avg_ms;
        recall_s << std::fixed << std::setprecision(3) << r.recall;
        scan_s  << (int)r.avg_scanned;
        std::cout << std::left
                  << std::setw(8)  << "IVF"
                  << std::setw(10) << r.nprobe
                  << std::setw(12) << ms_s.str()
                  << std::setw(14) << recall_s.str()
                  << std::setw(14) << scan_s.str()
                  << "\n";
    }
    std::cout << std::string(58, '-') << "\n\n";

    // -----------------------------------------------------------------------
    // Step 8: Quick summary analysis
    // -----------------------------------------------------------------------
    std::cout << " Summary:\n";
    std::cout << "  Insertion throughput : " << (int)ins_per_sec << " vec/sec\n";
    std::cout << "  BUILD time           : " << (int)build_ms << " ms\n";
    std::cout << "  BRUTE avg latency    : " << std::fixed << std::setprecision(3)
              << avg_brute_ms << " ms\n";
    if (!results_table.empty()) {
        auto& best = results_table.back(); // highest nprobe
        auto& fast = results_table.front(); // nprobe=1
        std::cout << "  IVF nprobe=1 recall  : " << std::setprecision(1)
                  << fast.recall * 100 << "%  scanned=" << (int)fast.avg_scanned << "\n";
        std::cout << "  IVF nprobe=" << best.nprobe << " recall : "
                  << std::setprecision(1) << best.recall * 100
                  << "%  scanned=" << (int)best.avg_scanned << "\n";
    }
    std::cout << "\n";

    send_line(fd, "QUIT");
    recv_line(fd);
    close(fd);
    return 0;
}
