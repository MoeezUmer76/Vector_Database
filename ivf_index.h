#pragma once
// =============================================================================
// ivf_index.h
// Inverted File (IVF) index built on top of VectorStore.
//
// Design:
//   - Centroids are stored in a flat array (K × D floats), owned here.
//   - Each cluster stores only the *indices* into VectorStore — no vector copy.
//   - k-means (Lloyd's algorithm) runs directly on VectorStore's flat array.
//
// Thread safety:
//   The caller (server.cpp) must hold g_mutex before calling build() or any
//   search/add method, exactly as with VectorStore.
// =============================================================================

#include <vector>
#include <cstdint>
#include <string>
#include "vector_store.h"

// Result of an IVF search — extends SearchResult with scan metadata
struct IVFSearchResult {
    float    distance;
    uint64_t id;
    int      vectors_scanned; // total vectors examined across probed clusters
};

class IVFIndex {
public:
    // ---------- Construction / Build ----------------------------------------

    // Build the IVF index from the current contents of `store`.
    // K = floor(sqrt(N));  max_iters = 50.
    // Returns a human-readable build summary (iterations, time, cluster sizes).
    std::string build(const VectorStore& store);

    // ---------- Search -------------------------------------------------------

    // IVF approximate k-NN.
    // nprobe: number of closest centroids to probe.
    // Returns results sorted by ascending distance, plus vectors_scanned count.
    std::vector<IVFSearchResult> search(
        const VectorStore& store,
        const std::vector<float>& query,
        int k,
        int nprobe) const;

    // ---------- Incremental ADD ----------------------------------------------

    // Assign a newly added vector (at store index `new_idx`) to the nearest
    // centroid and append it to that cluster list.
    // Call this AFTER VectorStore::add() and BEFORE releasing the mutex.
    void add_vector(const VectorStore& store, int new_idx);

    // ---------- State queries ------------------------------------------------

    bool is_built() const { return built_; }
    int  num_clusters() const { return K_; }

    // Returns a multiline string listing cluster sizes (for STATS)
    std::string cluster_stats() const;

    // --- Snapshot support ---------------------------------------------------

    // Raw accessors for serialization (snapshot.cpp only)
    const std::vector<float>&              get_centroids() const { return centroids_; }
    const std::vector<std::vector<int>>&   get_clusters()  const { return clusters_; }

    // Bulk-replace IVF state from pre-validated raw data.
    // Used exclusively by load_snapshot(); do NOT call from other code.
    void restore(int dim, int K,
                 std::vector<float>              centroids,
                 std::vector<std::vector<int>>   clusters);

private:
    // --- k-means internals ---------------------------------------------------

    // Initialise centroids by picking K random distinct vectors from store
    void init_centroids_random(const VectorStore& store, int K);

    // Assign every vector in store to its nearest centroid.
    // Returns true if any assignment changed (used as convergence test).
    bool assign_clusters(const VectorStore& store,
                         std::vector<int>& assignments) const;

    // Recompute centroid positions as the mean of assigned vectors.
    // Empty clusters keep their previous centroid (no crash, no NaN).
    void recompute_centroids(const VectorStore& store,
                             const std::vector<int>& assignments);

    // --- Distance helper (reuses VectorStore's sq_dist via pointer math) ----
    float sq_dist_to_centroid(const float* vec, int c_idx) const;

    // Find nearest centroid index for a query vector
    int nearest_centroid(const float* query) const;

    // --- Data ----------------------------------------------------------------
    bool built_ = false;
    int  K_     = 0;   // number of clusters
    int  dim_   = 0;   // vector dimension (copy from store for convenience)

    // Flat centroid array: centroids_[c * dim_ .. c * dim_ + dim_ - 1]
    std::vector<float> centroids_;

    // Inverted lists: clusters_[c] = sorted list of store indices in cluster c
    std::vector<std::vector<int>> clusters_;
};
