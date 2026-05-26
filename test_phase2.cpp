// =============================================================================
// test_phase2.cpp
// Verifies correctness of Phase 2 (k-means + IVF) without a network connection.
// =============================================================================

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>

#include "vector_store.h"
#include "ivf_index.h"
#include "command_parser.h"

static std::set<uint64_t> ids_of(const std::vector<SearchResult>& v) {
    std::set<uint64_t> s;
    for (auto& r : v) s.insert(r.id);
    return s;
}

static std::set<uint64_t> ids_of_ivf(const std::vector<IVFSearchResult>& v) {
    std::set<uint64_t> s;
    for (auto& r : v) s.insert(r.id);
    return s;
}

void test_kmeans_covers_all() {
    std::cout << "\n[TEST 1] k-means covers all vectors\n";
    VectorStore store(2);
    for (int i = 0; i < 16; ++i)
        store.add(i, {(float)i, (float)(i * 2)});

    IVFIndex ivf;
    std::string summary = ivf.build(store);
    std::cout << "  build summary: " << summary << "\n";

    assert(ivf.is_built());
    assert(ivf.num_clusters() == 4);

    std::string cs = ivf.cluster_stats();
    std::cout << "  cluster_stats: " << cs << "\n";

    std::cout << "  PASS\n";
}

void test_ivf_equals_brute_at_max_nprobe() {
    std::cout << "\n[TEST 2] IVF(nprobe=K) == BRUTE\n";
    VectorStore store(3);
    int id = 0;
    for (int x = 0; x < 5; ++x)
        for (int y = 0; y < 5; ++y)
            store.add(id++, {(float)x * 10.0f, (float)y * 10.0f, 0.0f});

    IVFIndex ivf;
    ivf.build(store);
    int K = ivf.num_clusters();

    std::vector<float> query = {3.0f, 3.0f, 0.0f};
    int k = 5;

    auto brute = store.search_brute(query, k);
    auto ivf_res = ivf.search(store, query, k, K);

    std::cout << "  BRUTE ids: ";
    for (auto& r : brute) std::cout << r.id << "(" << r.distance << ") ";
    std::cout << "\n";

    std::cout << "  IVF   ids: ";
    for (auto& r : ivf_res) std::cout << r.id << "(" << r.distance << ") ";
    std::cout << "  scanned=" << (ivf_res.empty() ? 0 : ivf_res[0].vectors_scanned) << "\n";

    assert(ids_of(brute) == ids_of_ivf(ivf_res));

    for (size_t i = 1; i < ivf_res.size(); ++i)
        assert(ivf_res[i].distance >= ivf_res[i-1].distance);

    std::cout << "  PASS\n";
}

void test_incremental_add() {
    std::cout << "\n[TEST 3] Incremental ADD after BUILD\n";
    VectorStore store(2);
    for (int i = 0; i < 9; ++i)
        store.add(i, {(float)i, 0.0f});

    IVFIndex ivf;
    ivf.build(store);

    auto before = ivf.cluster_stats();
    std::cout << "  before: " << before << "\n";

    store.add(100, {4.5f, 0.0f});
    ivf.add_vector(store, store.size() - 1);

    auto after = ivf.cluster_stats();
    std::cout << "  after:  " << after << "\n";

    std::vector<float> q = {4.5f, 0.0f};
    auto res = ivf.search(store, q, 1, ivf.num_clusters());
    assert(!res.empty());
    assert(res[0].id == 100);
    std::cout << "  nearest to (4.5,0): id=" << res[0].id
              << " dist=" << res[0].distance << "\n";

    std::cout << "  PASS\n";
}

void test_k_greater_than_n() {
    std::cout << "\n[TEST 4] k > N (should return N results, not crash)\n";
    VectorStore store(2);
    store.add(1, {0.0f, 0.0f});
    store.add(2, {1.0f, 1.0f});

    IVFIndex ivf;
    ivf.build(store);

    auto res = ivf.search(store, {0.5f, 0.5f}, 10, ivf.num_clusters());
    assert(res.size() == 2);
    std::cout << "  returned " << res.size() << " results (correct)\n";
    std::cout << "  PASS\n";
}

void test_parser_phase2() {
    std::cout << "\n[TEST 5] Parser: BUILD and SEARCH IVF\n";

    auto cmd_build = parse_command("BUILD", 3);
    assert(cmd_build.type == CommandType::BUILD);
    assert(cmd_build.error.empty());

    auto cmd_ivf = parse_command("SEARCH 1.0 2.0 3.0 5 IVF 3", 3);
    assert(cmd_ivf.type == CommandType::SEARCH);
    assert(cmd_ivf.method == SearchMethod::IVF);
    assert(cmd_ivf.k == 5);
    assert(cmd_ivf.nprobe == 3);
    assert(cmd_ivf.error.empty());

    auto cmd_bad = parse_command("SEARCH 1.0 2.0 3.0 5 IVF", 3);
    assert(!cmd_bad.error.empty());

    auto cmd_neg = parse_command("SEARCH 1.0 2.0 3.0 5 IVF -1", 3);
    assert(!cmd_neg.error.empty());

    auto cmd_brute = parse_command("SEARCH 1.0 2.0 3.0 5 BRUTE", 3);
    assert(cmd_brute.type == CommandType::SEARCH);
    assert(cmd_brute.method == SearchMethod::BRUTE);

    std::cout << "  PASS\n";
}

void test_result_ordering() {
    std::cout << "\n[TEST 6] Results sorted by ascending distance\n";
    VectorStore store(2);
    for (int i = 0; i < 16; ++i)
        store.add(i, {(float)(i * 3), (float)(i * 3)});

    IVFIndex ivf;
    ivf.build(store);

    std::vector<float> q = {10.0f, 10.0f};
    int k = 8;

    auto br = store.search_brute(q, k);
    for (size_t i = 1; i < br.size(); ++i)
        assert(br[i].distance >= br[i-1].distance);

    auto iv = ivf.search(store, q, k, ivf.num_clusters());
    for (size_t i = 1; i < iv.size(); ++i)
        assert(iv[i].distance >= iv[i-1].distance);

    std::cout << "  PASS\n";
}

int main() {
    std::cout << "=== Phase 2 Tests ===\n";
    test_kmeans_covers_all();
    test_ivf_equals_brute_at_max_nprobe();
    test_incremental_add();
    test_k_greater_than_n();
    test_parser_phase2();
    test_result_ordering();
    std::cout << "\n=== ALL TESTS PASSED ===\n";
    return 0;
}