// =============================================================================
// snapshot.cpp
// Binary persistence: save_snapshot() and load_snapshot()
//
// See snapshot.h for the full file format specification.
// =============================================================================

#include "snapshot.h"

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <sstream>
#include <vector>
#include <sys/stat.h>   // stat()

// ---------------------------------------------------------------------------
// File format constants
// ---------------------------------------------------------------------------
static const char    MAGIC[4]   = {'V', 'D', 'B', '1'};
static const uint32_t VERSION    = 1;

// ---------------------------------------------------------------------------
// Low-level write helpers — return false on any I/O error
// ---------------------------------------------------------------------------

static bool write_u32(FILE* f, uint32_t v) {
    return std::fwrite(&v, sizeof(v), 1, f) == 1;
}

static bool write_u64(FILE* f, uint64_t v) {
    return std::fwrite(&v, sizeof(v), 1, f) == 1;
}

// Write a contiguous block of floats directly from the pointer
static bool write_floats(FILE* f, const float* data, size_t count) {
    if (count == 0) return true;
    return std::fwrite(data, sizeof(float), count, f) == count;
}

static bool write_u32_array(FILE* f, const int* data, size_t count) {
    // Indices stored as uint32 on disk; store is int internally.
    // Write element-by-element to handle sign conversion cleanly.
    for (size_t i = 0; i < count; ++i) {
        uint32_t v = static_cast<uint32_t>(data[i]);
        if (std::fwrite(&v, sizeof(v), 1, f) != 1) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Low-level read helpers — return false on EOF or short read
// ---------------------------------------------------------------------------

static bool read_u32(FILE* f, uint32_t& v) {
    return std::fread(&v, sizeof(v), 1, f) == 1;
}

static bool read_u64(FILE* f, uint64_t& v) {
    return std::fread(&v, sizeof(v), 1, f) == 1;
}

static bool read_floats(FILE* f, float* data, size_t count) {
    if (count == 0) return true;
    return std::fread(data, sizeof(float), count, f) == count;
}

// ---------------------------------------------------------------------------
// save_snapshot
// ---------------------------------------------------------------------------
SnapshotResult save_snapshot(const VectorStore& store,
                              const IVFIndex&    ivf,
                              const std::string& path) {
    SnapshotResult res;

    const int      N   = store.size();
    const int      D   = store.dim();
    const int      K   = ivf.is_built() ? ivf.num_clusters() : 0;

    std::string tmp_path = path + ".tmp";

    FILE* f = std::fopen(tmp_path.c_str(), "wb");
    if (!f) {
        res.message = std::string("ERR cannot open tmp file for writing: ") + std::strerror(errno);
        return res;
    }

    // --- Header ---
    bool ok = true;
    ok &= (std::fwrite(MAGIC, 1, 4, f) == 4);
    ok &= write_u32(f, VERSION);
    ok &= write_u32(f, static_cast<uint32_t>(D));
    ok &= write_u32(f, static_cast<uint32_t>(N));
    ok &= write_u32(f, static_cast<uint32_t>(K));

    // --- IDs (int64 / uint64) ---
    for (int i = 0; i < N && ok; ++i)
        ok &= write_u64(f, store.get_id(i));

    // --- Vectors (flat float array, N*D floats) ---
    if (N > 0 && ok)
        ok &= write_floats(f, store.get_vector_ptr(0), static_cast<size_t>(N) * D);

    // --- Centroids (K*D floats, only if IVF is built) ---
    if (K > 0 && ok) {
        const auto& centroids = ivf.get_centroids();
        ok &= write_floats(f, centroids.data(), centroids.size());
    }

    // --- Cluster lists ---
    if (K > 0 && ok) {
        const auto& clusters = ivf.get_clusters();
        for (int c = 0; c < K && ok; ++c) {
            uint32_t sz = static_cast<uint32_t>(clusters[c].size());
            ok &= write_u32(f, sz);
            ok &= write_u32_array(f, clusters[c].data(), clusters[c].size());
        }
    }

    if (!ok) {
        std::fclose(f);
        std::remove(tmp_path.c_str());
        res.message = "ERR I/O error while writing snapshot";
        return res;
    }

    // Flush to OS buffers, then close
    if (std::fflush(f) != 0) {
        std::fclose(f);
        std::remove(tmp_path.c_str());
        res.message = std::string("ERR fflush failed: ") + std::strerror(errno);
        return res;
    }
    std::fclose(f);

    // --- Atomic rename ---
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        res.message = std::string("ERR rename failed: ") + std::strerror(errno);
        return res;
    }

    // Report file size
    struct stat st{};
    if (stat(path.c_str(), &st) == 0)
        res.file_bytes = static_cast<size_t>(st.st_size);

    res.ok       = true;
    res.vectors  = static_cast<uint32_t>(N);
    res.clusters = static_cast<uint32_t>(K);
    std::ostringstream oss;
    oss << "saved vectors=" << N << " clusters=" << K
        << " file_bytes=" << res.file_bytes;
    res.message = oss.str();
    return res;
}

// ---------------------------------------------------------------------------
// load_snapshot
// ---------------------------------------------------------------------------
SnapshotResult load_snapshot(VectorStore&       store,
                              IVFIndex&          ivf,
                              const std::string& path) {
    SnapshotResult res;

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        res.message = std::string("ERR cannot open snapshot: ") + std::strerror(errno);
        return res;
    }

    // --- Helper macro to bail out cleanly on read error ---
#define BAIL(msg) do { std::fclose(f); res.message = (msg); return res; } while(0)

    // --- Magic bytes ---
    char magic[4];
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, MAGIC, 4) != 0)
        BAIL("ERR invalid magic bytes — not a VecDB snapshot");

    // --- Version ---
    uint32_t version;
    if (!read_u32(f, version) || version != VERSION) {
        std::ostringstream e;
        e << "ERR unsupported snapshot version=" << version << " (expected " << VERSION << ")";
        BAIL(e.str());
    }

    // --- Dimension ---
    uint32_t file_dim;
    if (!read_u32(f, file_dim))  BAIL("ERR truncated header (dimension)");
    if (static_cast<int>(file_dim) != store.dim()) {
        std::ostringstream e;
        e << "ERR dimension mismatch: file=" << file_dim
          << " server=" << store.dim();
        BAIL(e.str());
    }

    const int D = store.dim();

    // --- Vector count ---
    uint32_t N;
    if (!read_u32(f, N))  BAIL("ERR truncated header (vector_count)");

    // --- Cluster count ---
    uint32_t K;
    if (!read_u32(f, K))  BAIL("ERR truncated header (cluster_count)");

    // --- IDs ---
    std::vector<uint64_t> ids(N);
    for (uint32_t i = 0; i < N; ++i) {
        if (!read_u64(f, ids[i])) BAIL("ERR truncated ids array");
    }

    // --- Vectors ---
    std::vector<float> vectors(static_cast<size_t>(N) * D);
    if (!read_floats(f, vectors.data(), vectors.size()))
        BAIL("ERR truncated vectors array");

    // --- Centroids (only if K > 0) ---
    std::vector<float> centroids;
    if (K > 0) {
        centroids.resize(static_cast<size_t>(K) * D);
        if (!read_floats(f, centroids.data(), centroids.size()))
            BAIL("ERR truncated centroids array");
    }

    // --- Cluster lists ---
    std::vector<std::vector<int>> clusters(K);
    for (uint32_t c = 0; c < K; ++c) {
        uint32_t sz;
        if (!read_u32(f, sz)) BAIL("ERR truncated cluster size");
        clusters[c].resize(sz);
        for (uint32_t j = 0; j < sz; ++j) {
            uint32_t idx;
            if (!read_u32(f, idx)) BAIL("ERR truncated cluster indices");
            // Validate index is in range
            if (idx >= N) BAIL("ERR cluster index out of range");
            clusters[c][j] = static_cast<int>(idx);
        }
    }

    std::fclose(f);
#undef BAIL

    // --- All data read and validated: now replace in-memory state ---
    store.restore(ids, vectors);
    if (K > 0)
        ivf.restore(D, static_cast<int>(K),
                    std::move(centroids), std::move(clusters));

    res.ok       = true;
    res.vectors  = N;
    res.clusters = K;
    std::ostringstream oss;
    oss << "loaded vectors=" << N << " clusters=" << K;
    res.message  = oss.str();
    return res;
}

// ---------------------------------------------------------------------------
bool snapshot_exists(const std::string& path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0;
}
