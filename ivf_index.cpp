// =============================================================================
// ivf_index.cpp
// k-means (Lloyd's algorithm) + IVF index build and search.
// =============================================================================

#include "ivf_index.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Helper: squared Euclidean distance between pointer `a` and centroid c_idx
// ---------------------------------------------------------------------------
float IVFIndex::sq_dist_to_centroid(const float* vec, int c_idx) const {
    const float* cen = centroids_.data() + (size_t)c_idx * dim_;
    float sum = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float diff = vec[i] - cen[i];
        sum += diff * diff;
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Find the index of the centroid nearest to `query`
// ---------------------------------------------------------------------------
int IVFIndex::nearest_centroid(const float* query) const {
    int    best_c = 0;
    float  best_d = std::numeric_limits<float>::max();
    for (int c = 0; c < K_; ++c) {
        float d = sq_dist_to_centroid(query, c);
        if (d < best_d) { best_d = d; best_c = c; }
    }
    return best_c;
}

// ---------------------------------------------------------------------------
// Randomly pick K distinct vectors from store as initial centroids
// ---------------------------------------------------------------------------
void IVFIndex::init_centroids_random(const VectorStore& store, int K) {
    int N = store.size();
    // Generate a shuffled index list and take the first K
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    // Use a fixed seed for reproducibility during testing; swap to random_device
    // for production use.
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    centroids_.resize((size_t)K * dim_);
    for (int c = 0; c < K; ++c) {
        const float* src = store.get_vector_ptr(indices[c]);
        float*       dst = centroids_.data() + (size_t)c * dim_;
        std::copy(src, src + dim_, dst);
    }
}

// ---------------------------------------------------------------------------
// Assign each vector to its nearest centroid.
// Returns true if ANY assignment changed (convergence test).
// ---------------------------------------------------------------------------
bool IVFIndex::assign_clusters(const VectorStore& store,
                                std::vector<int>& assignments) const {
    int  N       = store.size();
    bool changed = false;
    for (int i = 0; i < N; ++i) {
        int c = nearest_centroid(store.get_vector_ptr(i));
        if (c != assignments[i]) {
            assignments[i] = c;
            changed = true;
        }
    }
    return changed;
}

// ---------------------------------------------------------------------------
// Recompute each centroid as the mean of its assigned vectors.
// Empty clusters: keep the old centroid (safe, no NaN).
// ---------------------------------------------------------------------------
void IVFIndex::recompute_centroids(const VectorStore& store,
                                   const std::vector<int>& assignments) {
    int N = store.size();

    // Accumulator: sum of vectors per cluster, plus count
    std::vector<float> sums((size_t)K_ * dim_, 0.0f);
    std::vector<int>   counts(K_, 0);

    for (int i = 0; i < N; ++i) {
        int          c   = assignments[i];
        const float* vec = store.get_vector_ptr(i);
        float*       acc = sums.data() + (size_t)c * dim_;
        for (int d = 0; d < dim_; ++d) acc[d] += vec[d];
        counts[c]++;
    }

    for (int c = 0; c < K_; ++c) {
        if (counts[c] == 0) continue; // keep old centroid — avoid NaN
        float*       cen = centroids_.data() + (size_t)c * dim_;
        const float* acc = sums.data()       + (size_t)c * dim_;
        float inv = 1.0f / (float)counts[c];
        for (int d = 0; d < dim_; ++d) cen[d] = acc[d] * inv;
    }
}

// ---------------------------------------------------------------------------
// BUILD: run k-means and populate the inverted lists
// ---------------------------------------------------------------------------
std::string IVFIndex::build(const VectorStore& store) {
    int N = store.size();
    if (N == 0) throw std::runtime_error("Cannot build IVF index on empty store");

    dim_ = store.dim();
    K_   = std::max(1, (int)std::sqrt((double)N)); // K = floor(sqrt(N))

    std::cout << "[ivf] Building index: N=" << N
              << "  K=" << K_ << "  dim=" << dim_ << "\n";

    auto t_start = std::chrono::steady_clock::now();

    // --- k-means ---
    const int MAX_ITERS = 50;

    init_centroids_random(store, K_);

    // assignments[i] = cluster index for vector i; init to -1 (unassigned)
    std::vector<int> assignments(N, -1);

    int iter = 0;
    for (; iter < MAX_ITERS; ++iter) {
        bool changed = assign_clusters(store, assignments);

        // Debug: print cluster sizes every 10 iterations
        if (iter % 10 == 0 || !changed) {
            std::vector<int> sz(K_, 0);
            for (int a : assignments) sz[a]++;
            int min_sz = *std::min_element(sz.begin(), sz.end());
            int max_sz = *std::max_element(sz.begin(), sz.end());
            std::cout << "[ivf] iter=" << iter
                      << "  changed=" << changed
                      << "  cluster_sizes min=" << min_sz
                      << " max=" << max_sz << "\n";
        }

        recompute_centroids(store, assignments);

        if (!changed) {
            ++iter; // count completed iterations correctly
            break;
        }
    }

    // --- Populate inverted lists ---
    clusters_.assign(K_, {});
    for (int i = 0; i < N; ++i)
        clusters_[assignments[i]].push_back(i);

    built_ = true;

    auto t_end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Build summary string
    std::ostringstream oss;
    oss << "clusters=" << K_
        << " iterations=" << iter
        << " time_ms=" << (int)ms;
    return oss.str();
}

// ---------------------------------------------------------------------------
// SEARCH IVF
// ---------------------------------------------------------------------------
std::vector<IVFSearchResult> IVFIndex::search(
        const VectorStore& store,
        const std::vector<float>& query,
        int k,
        int nprobe) const {

    if (!built_) throw std::runtime_error("IVF index not built — run BUILD first");
    if ((int)query.size() != dim_)
        throw std::invalid_argument("Query dimension mismatch");

    // --- Step 1: find nprobe nearest centroids using a max-heap ----
    //   We want the nprobe *smallest* distances.
    //   Strategy: run max-heap of size nprobe over all K centroids.
    int effective_np = std::min(nprobe, K_);

    using CPair = std::pair<float, int>; // (dist, centroid_idx)
    std::priority_queue<CPair> centroid_heap;

    for (int c = 0; c < K_; ++c) {
        float d = sq_dist_to_centroid(query.data(), c);
        if ((int)centroid_heap.size() < effective_np) {
            centroid_heap.push({d, c});
        } else if (d < centroid_heap.top().first) {
            centroid_heap.pop();
            centroid_heap.push({d, c});
        }
    }

    // Collect the chosen centroid indices (order doesn't matter here)
    std::vector<int> probed_centroids;
    probed_centroids.reserve(effective_np);
    while (!centroid_heap.empty()) {
        probed_centroids.push_back(centroid_heap.top().second);
        centroid_heap.pop();
    }

    // --- Step 2: scan vectors in probed clusters, maintain top-k max-heap ---
    int effective_k = std::min(k, store.size());

    using RPair = std::pair<float, uint64_t>; // (dist, id)
    std::priority_queue<RPair> result_heap;   // max-heap, top = farthest

    int vectors_scanned = 0;

    for (int c : probed_centroids) {
        for (int idx : clusters_[c]) {
            const float* vec = store.get_vector_ptr(idx);
            float d = 0.0f;
            // Squared Euclidean distance: inline for speed
            for (int i = 0; i < dim_; ++i) {
                float diff = query[i] - vec[i];
                d += diff * diff;
            }
            ++vectors_scanned;

            if ((int)result_heap.size() < effective_k) {
                result_heap.push({d, store.get_id(idx)});
            } else if (d < result_heap.top().first) {
                result_heap.pop();
                result_heap.push({d, store.get_id(idx)});
            }
        }
    }

    // --- Step 3: drain heap and reverse to get ascending order ---
    std::vector<IVFSearchResult> results;
    results.reserve(result_heap.size());
    while (!result_heap.empty()) {
        results.push_back({result_heap.top().first,
                           result_heap.top().second,
                           vectors_scanned});
        result_heap.pop();
    }
    std::reverse(results.begin(), results.end());

    // Attach scanned count to every result row
    for (auto& r : results) r.vectors_scanned = vectors_scanned;
    return results;
}

// ---------------------------------------------------------------------------
// Incremental ADD: assign new vector to nearest centroid
// ---------------------------------------------------------------------------
void IVFIndex::add_vector(const VectorStore& store, int new_idx) {
    if (!built_) return; // no-op before BUILD
    int c = nearest_centroid(store.get_vector_ptr(new_idx));
    clusters_[c].push_back(new_idx);
    // Note: centroids are NOT recomputed (incremental mode)
}

// ---------------------------------------------------------------------------
// Snapshot support: bulk-replace IVF state (used by load_snapshot only)
// ---------------------------------------------------------------------------
void IVFIndex::restore(int dim, int K,
                        std::vector<float>            centroids,
                        std::vector<std::vector<int>> clusters) {
    dim_       = dim;
    K_         = K;
    centroids_ = std::move(centroids);
    clusters_  = std::move(clusters);
    built_     = true;
}

// ---------------------------------------------------------------------------
// Cluster stats for STATS command
// ---------------------------------------------------------------------------
std::string IVFIndex::cluster_stats() const {
    if (!built_) return "ivf=not_built";

    std::vector<int> sizes;
    sizes.reserve(K_);
    for (const auto& cl : clusters_) sizes.push_back((int)cl.size());

    int min_sz = *std::min_element(sizes.begin(), sizes.end());
    int max_sz = *std::max_element(sizes.begin(), sizes.end());
    double avg = (double)std::accumulate(sizes.begin(), sizes.end(), 0) / K_;

    std::ostringstream oss;
    oss << "ivf_clusters=" << K_
        << " cluster_min=" << min_sz
        << " cluster_max=" << max_sz
        << " cluster_avg=" << (int)avg;
    return oss.str();
}
