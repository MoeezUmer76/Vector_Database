#pragma once
// =============================================================================
// vector_store.h
// Core data structure: stores vectors in a flat array, supports brute-force kNN
// =============================================================================

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <string>
#include <queue>
#include <cmath>
#include <mutex>
#include <sstream>
#include <stdexcept>

// Represents one search result: (distance, id)
struct SearchResult {
    float  distance;
    uint64_t id;
};

class VectorStore {
public:
    // D = fixed dimension, set once at startup and never changed
    explicit VectorStore(int dimension);

    // --- Mutating operations (require external lock) ---

    // Add a vector with the given id.
    // Returns false if the id already exists.
    bool add(uint64_t id, const std::vector<float>& vec);

    // --- Read operations (require external lock) ---

    // Brute-force k-nearest-neighbour search.
    // Returns up to k results sorted by ascending distance.
    std::vector<SearchResult> search_brute(const std::vector<float>& query, int k) const;

    // Stats string
    std::string stats() const;

    // Dimension accessor
    int dim() const { return dim_; }

    // --- Read-only accessors used by IVFIndex (no copies) ---

    // Total number of stored vectors
    int size() const { return (int)ids_.size(); }

    // Raw pointer to vector i's floats (dim_ floats starting here)
    // Caller must hold the external mutex while using this pointer.
    const float* get_vector_ptr(int idx) const {
        return vectors_.data() + (size_t)idx * dim_;
    }

    // ID of the vector at internal index idx
    uint64_t get_id(int idx) const { return ids_[idx]; }

    // Squared Euclidean distance (public so IVFIndex can reuse it)
    float sq_dist(const float* a, const float* b) const;

    // Stats string; optionally append IVF-level info passed in from outside
    std::string stats(const std::string& extra = "") const;

    // --- Snapshot support ---------------------------------------------------

    // Bulk-replace store contents from pre-validated raw data.
    // Used exclusively by load_snapshot(); do NOT call from other code.
    // Precondition: dim matches dim_ (caller must validate).
    void restore(const std::vector<uint64_t>&  ids,
                 const std::vector<float>&     vectors);

private:
    int dim_;                                   // fixed vector dimension

    // Flat storage: vectors_[i * dim_ .. i * dim_ + dim_ - 1] = vector i
    std::vector<float>    vectors_;             // all raw floats
    std::vector<uint64_t> ids_;                 // ids_[i] = id of vector i
    std::unordered_map<uint64_t, int> id_to_idx_; // id → index in ids_
};
