// =============================================================================
// test_phase3.cpp
// Phase 3 tests: snapshot persistence, corruption handling, round-trip integrity
//
// Compile:
//   g++ -std=c++17 -O2 -o test_phase3 test_phase3.cpp \
//       vector_store.cpp ivf_index.cpp command_parser.cpp snapshot.cpp
// =============================================================================

#include <cassert>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include <random>
#include <numeric>

#include "vector_store.h"
#include "ivf_index.h"
#include "snapshot.h"
#include "command_parser.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void cleanup(const char* path) {
    std::remove(path);
    std::string tmp = std::string(path) + ".tmp";
    std::remove(tmp.c_str());
}

static bool floats_equal(const float* a, const float* b, int n, float eps = 1e-5f) {
    for (int i = 0; i < n; ++i)
        if (std::fabs(a[i] - b[i]) > eps) return false;
    return true;
}

// ============================================================================
// TEST 1: Empty database save/load round-trip
// ============================================================================
void test_empty_db_persistence() {
    std::cout << "\n[TEST 1] Empty database round-trip\n";
    const char* PATH = "test_empty.vdb";
    cleanup(PATH);

    VectorStore store(3);
    IVFIndex    ivf;

    SnapshotResult r = save_snapshot(store, ivf, PATH);
    assert(r.ok);
    assert(r.vectors == 0);
    assert(r.clusters == 0);
    std::cout << "  save: " << r.message << "\n";

    VectorStore store2(3);
    IVFIndex    ivf2;
    SnapshotResult r2 = load_snapshot(store2, ivf2, PATH);
    assert(r2.ok);
    assert(r2.vectors == 0);
    assert(r2.clusters == 0);
    assert(store2.size() == 0);
    assert(!ivf2.is_built());
    std::cout << "  load: " << r2.message << "\n";

    cleanup(PATH);
    std::cout << "  PASS\n";
}

// ============================================================================
// TEST 2: Full round-trip without IVF
// ============================================================================
void test_vectors_only_round_trip() {
    std::cout << "\n[TEST 2] Vectors-only round-trip (no IVF)\n";
    const char* PATH = "test_vecs.vdb";
    cleanup(PATH);

    const int D = 4, N = 10;
    VectorStore store(D);
    for (int i = 0; i < N; ++i)
        store.add(i, {(float)i, (float)(i*2), (float)(i*3), (float)(i*4)});

    IVFIndex ivf; // not built
    auto r = save_snapshot(store, ivf, PATH);
    assert(r.ok);
    assert(r.vectors == N);
    assert(r.clusters == 0);
    std::cout << "  save: " << r.message << "\n";

    VectorStore store2(D);
    IVFIndex    ivf2;
    auto r2 = load_snapshot(store2, ivf2, PATH);
    assert(r2.ok);
    assert(store2.size() == N);
    assert(!ivf2.is_built());

    // Verify every vector and ID
    for (int i = 0; i < N; ++i) {
        assert(store2.get_id(i) == (uint64_t)i);
        const float* v = store2.get_vector_ptr(i);
        assert(floats_equal(v, store.get_vector_ptr(i), D));
    }

    cleanup(PATH);
    std::cout << "  PASS\n";
}

// ============================================================================
// TEST 3: Full round-trip WITH IVF — centroids and cluster lists preserved
// ============================================================================
void test_full_round_trip_with_ivf() {
    std::cout << "\n[TEST 3] Full round-trip with IVF (centroids + cluster lists)\n";
    const char* PATH = "test_ivf.vdb";
    cleanup(PATH);

    const int D = 3, N = 25;
    VectorStore store(D);
    for (int i = 0; i < N; ++i)
        store.add(i, {(float)(i % 5) * 10.0f,
                      (float)(i / 5) * 10.0f,
                      (float)i * 0.1f});

    IVFIndex ivf;
    ivf.build(store);
    int K = ivf.num_clusters();
    std::cout << "  built K=" << K << " clusters\n";

    auto r = save_snapshot(store, ivf, PATH);
    assert(r.ok);
    assert(r.clusters == (uint32_t)K);
    std::cout << "  save: " << r.message << "\n";

    // Load into fresh objects
    VectorStore store2(D);
    IVFIndex    ivf2;
    auto r2 = load_snapshot(store2, ivf2, PATH);
    assert(r2.ok);
    assert(ivf2.is_built());
    assert(ivf2.num_clusters() == K);
    std::cout << "  load: " << r2.message << "\n";

    // Verify centroids match
    const auto& c1 = ivf.get_centroids();
    const auto& c2 = ivf2.get_centroids();
    assert(c1.size() == c2.size());
    assert(floats_equal(c1.data(), c2.data(), (int)c1.size()));

    // Verify cluster lists match
    const auto& cl1 = ivf.get_clusters();
    const auto& cl2 = ivf2.get_clusters();
    assert(cl1.size() == cl2.size());
    for (int c = 0; c < K; ++c) {
        assert(cl1[c].size() == cl2[c].size());
        for (size_t j = 0; j < cl1[c].size(); ++j)
            assert(cl1[c][j] == cl2[c][j]);
    }

    // Verify search produces identical results
    std::vector<float> query = {5.0f, 15.0f, 0.5f};
    auto res1 = ivf.search(store,   query, 5, K);
    auto res2 = ivf2.search(store2, query, 5, K);
    assert(res1.size() == res2.size());
    for (size_t i = 0; i < res1.size(); ++i) {
        assert(res1[i].id == res2[i].id);
        assert(std::fabs(res1[i].distance - res2[i].distance) < 1e-4f);
    }
    std::cout << "  Search results identical after restore\n";

    cleanup(PATH);
    std::cout << "  PASS\n";
}

// ============================================================================
// TEST 4: Corrupted snapshot handling — must not crash or corrupt state
// ============================================================================
void test_corrupted_snapshot() {
    std::cout << "\n[TEST 4] Corrupted snapshot — safe rejection\n";

    // 4a: File does not exist
    {
        VectorStore s(3); IVFIndex i;
        auto r = load_snapshot(s, i, "nonexistent_file.vdb");
        assert(!r.ok);
        std::cout << "  missing file: " << r.message << "\n";
    }

    // 4b: Bad magic bytes
    {
        const char* PATH = "test_corrupt_magic.vdb";
        {
            std::ofstream f(PATH, std::ios::binary);
            f.write("XYZW", 4);
            uint32_t v = 1; f.write((char*)&v, 4);
        }
        VectorStore s(3); IVFIndex i;
        auto r = load_snapshot(s, i, PATH);
        assert(!r.ok);
        assert(s.size() == 0); // state unchanged
        std::cout << "  bad magic: " << r.message << "\n";
        cleanup(PATH);
    }

    // 4c: Wrong version
    {
        const char* PATH = "test_corrupt_ver.vdb";
        {
            std::ofstream f(PATH, std::ios::binary);
            f.write("VDB1", 4);
            uint32_t v = 99; f.write((char*)&v, 4);
        }
        VectorStore s(3); IVFIndex i;
        auto r = load_snapshot(s, i, PATH);
        assert(!r.ok);
        std::cout << "  wrong version: " << r.message << "\n";
        cleanup(PATH);
    }

    // 4d: Dimension mismatch
    {
        const char* PATH = "test_corrupt_dim.vdb";
        // Build and save with D=3, load with D=5 store
        VectorStore s3(3); s3.add(1, {1.0f, 2.0f, 3.0f});
        IVFIndex i3;
        save_snapshot(s3, i3, PATH);
        VectorStore s5(5); IVFIndex i5;
        auto r = load_snapshot(s5, i5, PATH);
        assert(!r.ok);
        assert(s5.size() == 0); // unchanged
        std::cout << "  dim mismatch: " << r.message << "\n";
        cleanup(PATH);
    }

    // 4e: Truncated file (partial vectors)
    {
        const char* PATH = "test_corrupt_trunc.vdb";
        // Write valid header but truncate mid-vectors
        {
            std::ofstream f(PATH, std::ios::binary);
            f.write("VDB1", 4);
            uint32_t v = 1; f.write((char*)&v, 4);
            uint32_t d = 3; f.write((char*)&d, 4);
            uint32_t n = 100; f.write((char*)&n, 4);
            uint32_t k = 0;   f.write((char*)&k, 4);
            // Only write 5 IDs (truncated, should be 100)
            for (int i = 0; i < 5; ++i) {
                uint64_t id = i; f.write((char*)&id, 8);
            }
        }
        VectorStore s(3); IVFIndex i;
        auto r = load_snapshot(s, i, PATH);
        assert(!r.ok);
        assert(s.size() == 0); // unchanged
        std::cout << "  truncated: " << r.message << "\n";
        cleanup(PATH);
    }

    std::cout << "  PASS\n";
}

// ============================================================================
// TEST 5: IVF search correctness after restore (IVF(nprobe=K) == BRUTE)
// ============================================================================
void test_ivf_correctness_after_restore() {
    std::cout << "\n[TEST 5] IVF correctness after snapshot restore\n";
    const char* PATH = "test_ivf_correct.vdb";
    cleanup(PATH);

    const int D = 8, N = 64;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    VectorStore store(D);
    for (int i = 0; i < N; ++i) {
        std::vector<float> v(D);
        for (float& x : v) x = dist(rng);
        store.add(i, v);
    }

    IVFIndex ivf;
    ivf.build(store);
    int K = ivf.num_clusters();

    save_snapshot(store, ivf, PATH);

    // Restore
    VectorStore s2(D); IVFIndex i2;
    auto r = load_snapshot(s2, i2, PATH);
    assert(r.ok);

    // For 10 random queries: IVF(nprobe=K) must match BRUTE
    for (int q = 0; q < 10; ++q) {
        std::vector<float> query(D);
        for (float& x : query) x = dist(rng);

        auto brute = s2.search_brute(query, 5);
        auto ivfr  = i2.search(s2, query, 5, K);

        std::set<uint64_t> brute_ids, ivf_ids;
        for (auto& r : brute) brute_ids.insert(r.id);
        for (auto& r : ivfr)  ivf_ids.insert(r.id);
        assert(brute_ids == ivf_ids);
    }

    cleanup(PATH);
    std::cout << "  IVF(nprobe=K) == BRUTE for all test queries after restore\n";
    std::cout << "  PASS\n";
}

// ============================================================================
// TEST 6: Incremental ADD after restore works correctly
// ============================================================================
void test_incremental_add_after_restore() {
    std::cout << "\n[TEST 6] Incremental ADD after snapshot restore\n";
    const char* PATH = "test_incr.vdb";
    cleanup(PATH);

    VectorStore store(2);
    for (int i = 0; i < 9; ++i)
        store.add(i, {(float)i, (float)i});

    IVFIndex ivf;
    ivf.build(store);
    save_snapshot(store, ivf, PATH);

    // Restore and add a new vector
    VectorStore s2(2); IVFIndex i2;
    auto r = load_snapshot(s2, i2, PATH);
    assert(r.ok);

    int prev_sz = s2.size();
    s2.add(999, {4.5f, 4.5f});
    i2.add_vector(s2, prev_sz);

    // New vector must be findable
    std::vector<float> q = {4.5f, 4.5f};
    auto res = i2.search(s2, q, 1, i2.num_clusters());
    assert(!res.empty());
    assert(res[0].id == 999);
    std::cout << "  found newly added id=999 after restore\n";

    cleanup(PATH);
    std::cout << "  PASS\n";
}

// ============================================================================
// TEST 7: Command parser — SAVE and LOAD
// ============================================================================
void test_parser_save_load() {
    std::cout << "\n[TEST 7] Parser: SAVE and LOAD commands\n";

    auto c1 = parse_command("SAVE", 3);
    assert(c1.type == CommandType::SAVE);
    assert(c1.error.empty());

    auto c2 = parse_command("LOAD", 3);
    assert(c2.type == CommandType::LOAD);
    assert(c2.error.empty());

    auto c3 = parse_command("SAVE extra", 3);
    assert(!c3.error.empty()); // extra args rejected

    auto c4 = parse_command("LOAD extra", 3);
    assert(!c4.error.empty());

    std::cout << "  PASS\n";
}

// ============================================================================
// TEST 8: Large snapshot round-trip (stress test)
// ============================================================================
void test_large_snapshot() {
    std::cout << "\n[TEST 8] Large snapshot (1000 vectors, D=32)\n";
    const char* PATH = "test_large.vdb";
    cleanup(PATH);

    const int D = 32, N = 1000;
    std::mt19937 rng(99);
    std::normal_distribution<float> dist;

    VectorStore store(D);
    for (int i = 0; i < N; ++i) {
        std::vector<float> v(D);
        for (float& x : v) x = dist(rng);
        store.add(i * 7 + 3, v); // non-trivial IDs
    }

    IVFIndex ivf;
    ivf.build(store);
    int K = ivf.num_clusters();

    auto r = save_snapshot(store, ivf, PATH);
    assert(r.ok);
    std::cout << "  saved: " << r.message
              << "  file_bytes=" << r.file_bytes << "\n";

    VectorStore s2(D); IVFIndex i2;
    auto r2 = load_snapshot(s2, i2, PATH);
    assert(r2.ok);
    assert(s2.size() == N);
    assert(i2.num_clusters() == K);

    // Spot-check: all IDs present
    for (int i = 0; i < N; ++i)
        assert(s2.get_id(i) == (uint64_t)(i * 7 + 3));

    // Spot-check: vectors match
    for (int i = 0; i < N; ++i)
        assert(floats_equal(store.get_vector_ptr(i), s2.get_vector_ptr(i), D));

    cleanup(PATH);
    std::cout << "  All IDs and vectors match after large round-trip\n";
    std::cout << "  PASS\n";
}

// ============================================================================
int main() {
    std::cout << "=== Phase 3 Tests ===\n";
    test_empty_db_persistence();
    test_vectors_only_round_trip();
    test_full_round_trip_with_ivf();
    test_corrupted_snapshot();
    test_ivf_correctness_after_restore();
    test_incremental_add_after_restore();
    test_parser_save_load();
    test_large_snapshot();
    std::cout << "\n=== ALL 8 TESTS PASSED ===\n";
    return 0;
}
