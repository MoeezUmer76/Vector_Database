// =============================================================================
// client.cpp - COMPLETELY FIXED VERSION
// Properly reads ALL lines of multi-line responses
// =============================================================================

#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cctype>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// recv_line: read one '\n'-terminated line from fd.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// send_line: send a string + '\n' to fd.
// ---------------------------------------------------------------------------
static bool send_line(int fd, const std::string& msg) {
    std::string out = msg + "\n";
    ssize_t sent = send(fd, out.c_str(), out.size(), MSG_NOSIGNAL);
    return sent == (ssize_t)out.size();
}

// ---------------------------------------------------------------------------
// recv_response: read ALL lines until response is complete
// 
// CRITICAL FIX: For STATS, keep reading until we see "cluster sizes"
// Don't stop at "index built" because cluster sizes comes AFTER
// ---------------------------------------------------------------------------
static void recv_response(int fd) {
    std::string first_line = recv_line(fd);
    if (first_line.empty()) {
        std::cout << "[disconnected]\n";
        return;
    }
    
    std::cout << first_line << "\n";
    
    // 1. SEARCH response? (first line starts with digit)
    if (!first_line.empty() && std::isdigit(first_line[0])) {
        while (true) {
            std::string line = recv_line(fd);
            if (line.empty()) break;
            std::cout << line << "\n";
            if (!line.empty() && line[0] == '(') {
                break;  // Found summary line, SEARCH complete
            }
        }
        return;
    }
    
    // 2. BUILD response? (first line is "Building IVF index ...")
    if (first_line.find("Building") != std::string::npos) {
        while (true) {
            std::string line = recv_line(fd);
            if (line.empty()) break;
            std::cout << line << "\n";
            if (line == "OK") {
                break;  // BUILD complete
            }
        }
        return;
    }
    
    // 3. STATS response? (first line is "dimension : X")
    // CRITICAL FIX: Keep reading until we see "cluster sizes"
    // NOT "index built" because cluster sizes comes after!
    if (first_line.find("dimension") != std::string::npos) {
        while (true) {
            std::string line = recv_line(fd);
            if (line.empty()) break;
            std::cout << line << "\n";
            // Only stop after "cluster sizes" OR "index built : no"
            if (line.find("cluster sizes") != std::string::npos || 
                line.find("index built : no") != std::string::npos) {
                break;  // STATS complete
            }
        }
        return;
    }
    
    // 4. SAVE response? (first line starts with "Saved")
    if (first_line.find("Saved") != std::string::npos) {
        std::string line = recv_line(fd);
        if (!line.empty()) {
            std::cout << line << "\n";
        }
        return;
    }
    
    // 5. Otherwise single-line response (OK, ERR, etc)
    return;
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <host> <port>\n";
        return 1;
    }

    const char* host = argv[1];
    int port = std::atoi(argv[2]);

    hostent* he = gethostbyname(host);
    if (!he) {
        std::cerr << "ERR: cannot resolve host '" << host << "'\n";
        return 1;
    }

    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) { perror("socket"); return 1; }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons((uint16_t)port);
    std::memcpy(&server_addr.sin_addr, he->h_addr_list[0], he->h_length);

    if (connect(sock_fd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect");
        return 1;
    }

    std::cout << "[client] connected to " << host << ":" << port << "\n";

    recv_response(sock_fd);

    while (true) {
        std::cout << "> ";
        std::cout.flush();

        std::string line;
        if (!std::getline(std::cin, line)) break;

        if (line.empty()) continue;

        if (!send_line(sock_fd, line)) {
            std::cout << "[server closed connection]\n";
            break;
        }

        recv_response(sock_fd);

        std::string upper;
        for (char c : line) upper += (char)std::toupper((unsigned char)c);
        if (upper.find("QUIT") == 0) break;
    }

    close(sock_fd);
    std::cout << "[client] disconnected\n";
    return 0;
}