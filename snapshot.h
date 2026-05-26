#pragma once
// =============================================================================
// snapshot.h
// Binary snapshot persistence for VecDB.
//
// File format (all values little-endian):
//
//   Offset  Size  Field
//   ------  ----  -----
//   0       4     Magic bytes: 'V','D','B','1'
//   4       4     uint32  version  (currently 1)
//   8       4     uint32  dimension D
//   12      4     uint32  vector_count N
//   16      4     uint32  cluster_count K  (0 = no IVF index)
//   20      8*N   int64   ids[N]
//   20+8N   4*N*D float   vectors[N * D]    (row-major: vector i at [i*D])
//   --      4*K*D float   centroids[K * D]  (only if K > 0)
//   --      for each cluster c in [0,K):
//             4     uint32  cluster_size
//             4*sz  uint32  indices[cluster_size]
//
// Write pattern: write to <path>.tmp, flush, close, then rename to <path>.
// This ensures the on-disk file is always either the old complete snapshot
// or the new complete snapshot — never a torn partial write.
// =============================================================================

#include <string>
#include "vector_store.h"
#include "ivf_index.h"

// Default snapshot filename
inline constexpr const char* SNAPSHOT_FILENAME = "snapshot.vdb";
inline constexpr const char* SNAPSHOT_TMP      = "snapshot.vdb.tmp";

// Result returned by save_snapshot() / load_snapshot()
struct SnapshotResult {
    bool        ok      = false;
    std::string message;        // human-readable description or error
    uint32_t    vectors = 0;    // how many vectors were saved/loaded
    uint32_t    clusters = 0;   // how many clusters were saved/loaded
    size_t      file_bytes = 0; // file size in bytes (save only)
};

// Save the current state of `store` and `ivf` to `path`.
// The IVF index is saved only if ivf.is_built() == true.
// Uses temp-file-and-rename for atomicity.
SnapshotResult save_snapshot(const VectorStore& store,
                             const IVFIndex&    ivf,
                             const std::string& path = SNAPSHOT_FILENAME);

// Load a snapshot from `path` into `store` and `ivf`.
// `store` must already have the correct dimension (validated against file).
// On success, `store` and `ivf` are completely replaced.
// On any error, `store` and `ivf` are left UNCHANGED.
SnapshotResult load_snapshot(VectorStore&       store,
                             IVFIndex&          ivf,
                             const std::string& path = SNAPSHOT_FILENAME);

// Returns true if the snapshot file exists at `path`
bool snapshot_exists(const std::string& path = SNAPSHOT_FILENAME);
