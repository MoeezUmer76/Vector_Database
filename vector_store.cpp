// =============================================================================
// vector_store.cpp
// =============================================================================

#include "vector_store.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

// ---------------------------------------------------------------------------
VectorStore::VectorStore(int dimension) : dim_(dimension) {
    if (dimension <= 0)
        throw std::invalid_argument("Dimension must be > 0");
}

// ---------------------------------------------------------------------------
bool VectorStore::add(uint64_t id, const std::vector<float>& vec) {
    // Reject wrong-dimension vectors
    if ((int)vec.size() != dim_)
        throw std::invalid_argument("Vector dimension mismatch");

    // Reject duplicate IDs
    if (id_to_idx_.count(id))
        return false;

    // Record new index before inserting
    int new_idx = (int)ids_.size();

    // Append id and all floats to their respective arrays
    ids_.push_back(id);
    vectors_.insert(vectors_.end(), vec.begin(), vec.end());
    id_to_idx_[id] = new_idx;

    return true;
}

// ---------------------------------------------------------------------------
// Brute-force kNN using a max-heap of size k.
// Invariant: heap always holds the k *closest* points seen so far.
// ---------------------------------------------------------------------------
std::vector<SearchResult> VectorStore::search_brute(
        const std::vector<float>& query, int k) const {

    if ((int)query.size() != dim_)
        throw std::invalid_argument("Query dimension mismatch");

    int n = (int)ids_.size();
    if (n == 0 || k <= 0)
        return {};

    // Cap k at the number of stored vectors
    int effective_k = std::min(k, n);

    // Max-heap: top element is the *farthest* among the k candidates
    // pair<distance, id>  –  default comparator gives max-heap on first element
    using Pair = std::pair<float, uint64_t>;
    std::priority_queue<Pair> heap;

    for (int i = 0; i < n; ++i) {
        const float* v = vectors_.data() + (size_t)i * dim_;
        float d = sq_dist(query.data(), v);

        if ((int)heap.size() < effective_k) {
            heap.push({d, ids_[i]});
        } else if (d < heap.top().first) {
            // New point is closer than the farthest in heap → replace it
            heap.pop();
            heap.push({d, ids_[i]});
        }
    }

    // Drain the heap; results come out in descending distance order
    std::vector<SearchResult> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.push_back({heap.top().first, heap.top().second});
        heap.pop();
    }

    // Reverse to get ascending distance order (closest first)
    std::reverse(results.begin(), results.end());
    return results;
}

// ---------------------------------------------------------------------------
std::string VectorStore::stats(const std::string& extra) const {
    std::ostringstream oss;
    oss << "vectors=" << ids_.size()
        << " dimension=" << dim_
        << " memory_bytes=" << (vectors_.size() * sizeof(float));
    if (!extra.empty()) oss << " " << extra;
    return oss.str();
}

// ---------------------------------------------------------------------------
// Snapshot support: bulk-replace store contents (used by load_snapshot only)
// ---------------------------------------------------------------------------
void VectorStore::restore(const std::vector<uint64_t>& ids,
                           const std::vector<float>&    vectors) {
    ids_       = ids;
    vectors_   = vectors;
    id_to_idx_.clear();
    id_to_idx_.reserve(ids.size());
    for (int i = 0; i < (int)ids.size(); ++i)
        id_to_idx_[ids[i]] = i;
}
// ---------------------------------------------------------------------------
float VectorStore::sq_dist(const float* a, const float* b) const {
    float sum = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
