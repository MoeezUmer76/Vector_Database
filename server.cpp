// =============================================================================
// server.cpp - CORRECTED VERSION
// Multi-client TCP server for the Vector Database Engine (Phase 1+2+3)
//
// Protocol responses now match manual Section 3 exactly
// =============================================================================

#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <iomanip>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "vector_store.h"
#include "command_parser.h"
#include "ivf_index.h"
#include "snapshot.h"

// ---------------------------------------------------------------------------
// Globals shared across threads
// ---------------------------------------------------------------------------
static VectorStore*       g_store  = nullptr;
static IVFIndex*          g_ivf    = nullptr;
static std::mutex         g_mutex;
static std::atomic<int>   g_client_count{0};

// ---------------------------------------------------------------------------
// send_line: write a response string followed by '\n' to a socket fd.
// ---------------------------------------------------------------------------
static bool send_line(int fd, const std::string& msg) {
    std::string out = msg + "\n";
    ssize_t sent = send(fd, out.c_str(), out.size(), MSG_NOSIGNAL);
    return sent == (ssize_t)out.size();
}

// ---------------------------------------------------------------------------
// handle_client: runs in its own thread, one per connected client.
// ---------------------------------------------------------------------------
static void handle_client(int client_fd, std::string peer_addr) {
    int cid = ++g_client_count;
    std::cout << "[server] client #" << cid << " connected from " << peer_addr << "\n";

    send_line(client_fd, "OK");

    char buf[4096];
    std::string leftover;

    while (true) {
        ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) break;

        buf[n] = '\0';
        leftover += buf;

        size_t pos;
        while ((pos = leftover.find('\n')) != std::string::npos) {
            std::string line = leftover.substr(0, pos);
            leftover.erase(0, pos + 1);

            if (!line.empty() && line.back() == '\r')
                line.pop_back();

            Command cmd = parse_command(line, g_store->dim());

            if (!cmd.error.empty()) {
                send_line(client_fd, cmd.error);
                continue;
            }

            switch (cmd.type) {

            case CommandType::QUIT:
                send_line(client_fd, "OK");
                goto disconnect;

            case CommandType::STATS: {
                std::lock_guard<std::mutex> lock(g_mutex);
                
                std::ostringstream oss;
                oss << "dimension : " << g_store->dim() << "\n"
                    << "total vectors : " << g_store->size() << "\n";
                
                if (g_ivf->is_built()) {
                    oss << "index built : yes\n"
                        << "clusters : " << g_ivf->num_clusters() << "\n"
                        << "cluster sizes : ";
                    
                    // Get cluster sizes
                    const auto& clusters = g_ivf->get_clusters();
                    for (size_t i = 0; i < clusters.size(); ++i) {
                        if (i > 0) oss << " , ";
                        oss << clusters[i].size();
                    }
                } else {
                    oss << "index built : no";
                }
                
                send_line(client_fd, oss.str());
                break;
            }

            case CommandType::ADD: {
                std::lock_guard<std::mutex> lock(g_mutex);
                int prev_size = g_store->size();
                bool ok = g_store->add(cmd.id, cmd.floats);
                if (ok) {
                    if (g_ivf->is_built())
                        g_ivf->add_vector(*g_store, prev_size);
                    send_line(client_fd, "OK");
                } else {
                    send_line(client_fd, "ERR");
                }
                break;
            }

            case CommandType::BUILD: {
                send_line(client_fd, "Building IVF index ...");
                
                std::string summary;
                int vectors_count, clusters_count;
                double build_time_s;
                
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    try {
                        auto t_start = std::chrono::high_resolution_clock::now();
                        summary = g_ivf->build(*g_store);
                        auto t_end = std::chrono::high_resolution_clock::now();
                        
                        build_time_s = std::chrono::duration<double>(t_end - t_start).count();
                        vectors_count = g_store->size();
                        clusters_count = g_ivf->num_clusters();
                    } catch (const std::exception& e) {
                        send_line(client_fd, std::string("ERR ") + e.what());
                        break;
                    }
                }
                
                std::ostringstream oss;
                oss << "vectors : " << vectors_count << "\n"
                    << "clusters : " << clusters_count << "\n"
                    << "iterations : " << std::stoi(summary.substr(summary.find("iterations=") + 11)) << "\n"
                    << std::fixed << std::setprecision(3)
                    << "done in " << build_time_s << " s .";
                
                send_line(client_fd, oss.str());
                send_line(client_fd, "OK");
                break;
            }

            case CommandType::SEARCH: {
                if (cmd.method == SearchMethod::BRUTE) {
                    std::vector<SearchResult> results;
                    {
                        std::lock_guard<std::mutex> lock(g_mutex);
                        results = g_store->search_brute(cmd.floats, cmd.k);
                    }
                    
                    // Output each result: id distance v_1 v_2 ... v_D
                    for (const auto& r : results) {
                        std::ostringstream row;
                        row << r.id << " " << std::fixed << std::setprecision(2) << r.distance;
                        
                        const float* vec = nullptr;
                        {
                            std::lock_guard<std::mutex> lock(g_mutex);
                            for (int i = 0; i < g_store->size(); ++i) {
                                if (g_store->get_id(i) == r.id) {
                                    vec = g_store->get_vector_ptr(i);
                                    break;
                                }
                            }
                        }
                        
                        if (vec) {
                            for (int i = 0; i < g_store->dim(); ++i) {
                                row << " " << std::fixed << std::setprecision(2) << vec[i];
                            }
                        }
                        send_line(client_fd, row.str());
                    }
                    
                    std::ostringstream summary;
                    summary << "(" << results.size() << " results , mode = BRUTE , scanned " 
                            << g_store->size() << ")";
                    send_line(client_fd, summary.str());
                    
                } else {
                    // IVF search
                    if (!g_ivf->is_built()) {
                        send_line(client_fd, "ERR");
                        break;
                    }
                    
                    std::vector<IVFSearchResult> results;
                    try {
                        std::lock_guard<std::mutex> lock(g_mutex);
                        results = g_ivf->search(*g_store, cmd.floats, cmd.k, cmd.nprobe);
                    } catch (const std::exception& e) {
                        send_line(client_fd, "ERR");
                        break;
                    }
                    
                    // Output each result: id distance v_1 v_2 ... v_D
                    for (const auto& r : results) {
                        std::ostringstream row;
                        row << r.id << " " << std::fixed << std::setprecision(2) << r.distance;
                        
                        const float* vec = nullptr;
                        {
                            std::lock_guard<std::mutex> lock(g_mutex);
                            for (int i = 0; i < g_store->size(); ++i) {
                                if (g_store->get_id(i) == r.id) {
                                    vec = g_store->get_vector_ptr(i);
                                    break;
                                }
                            }
                        }
                        
                        if (vec) {
                            for (int i = 0; i < g_store->dim(); ++i) {
                                row << " " << std::fixed << std::setprecision(2) << vec[i];
                            }
                        }
                        send_line(client_fd, row.str());
                    }
                    
                    int scanned = results.empty() ? 0 : results[0].vectors_scanned;
                    std::ostringstream summary;
                    summary << "(" << results.size() << " results , mode = IVF , nprobe = " 
                            << cmd.nprobe << " , scanned " << scanned << ")";
                    send_line(client_fd, summary.str());
                }
                break;
            }

            case CommandType::SAVE: {
                SnapshotResult r;
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    r = save_snapshot(*g_store, *g_ivf);
                }
                if (r.ok) {
                    std::ostringstream oss;
                    oss << "Saved " << r.vectors << " vectors and IVF index to snapshot . vdb "
                        << "(" << r.file_bytes << " bytes ) .";
                    send_line(client_fd, oss.str());
                    send_line(client_fd, "OK");
                } else {
                    send_line(client_fd, "ERR");
                }
                break;
            }

            case CommandType::LOAD: {
                SnapshotResult r;
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    r = load_snapshot(*g_store, *g_ivf);
                }
                if (r.ok) {
                    send_line(client_fd, "OK");
                } else {
                    send_line(client_fd, "ERR");
                }
                break;
            }

            default:
                send_line(client_fd, "ERR");
            }
        }
    }

disconnect:
    close(client_fd);
    std::cout << "[server] client #" << cid << " disconnected\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <port> <dimension>\n";
        return 1;
    }

    int port = std::atoi(argv[1]);
    int dim  = std::atoi(argv[2]);

    if (port <= 0 || port > 65535) {
        std::cerr << "ERR: port must be in [1, 65535]\n";
        return 1;
    }
    if (dim <= 0) {
        std::cerr << "ERR: dimension must be > 0\n";
        return 1;
    }

    VectorStore store(dim);
    IVFIndex    ivf;
    g_store = &store;
    g_ivf   = &ivf;

    if (snapshot_exists()) {
        std::cout << "[server] Found snapshot.vdb — loading...\n";
        SnapshotResult r = load_snapshot(store, ivf);
        if (r.ok) {
            std::cout << "[server] Auto-load: " << r.message << "\n";
        } else {
            std::cout << "[server] WARNING: auto-load failed: " << r.message
                      << " — starting with empty database\n";
        }
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t)port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }

    if (listen(server_fd, 16) < 0) {
        perror("listen");
        return 1;
    }

    std::cout << "[server] VecDB listening on port " << port
              << "  dimension=" << dim << "\n";
    std::cout << "[server] Ctrl-C to stop\n";

    while (true) {
        sockaddr_in client_addr{};
        socklen_t   client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            perror("accept");
            continue;
        }

        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, ip_str, sizeof(ip_str));
        std::string peer = std::string(ip_str) + ":" +
                           std::to_string(ntohs(client_addr.sin_port));

        std::thread(handle_client, client_fd, peer).detach();
    }

    close(server_fd);
    return 0;
}
