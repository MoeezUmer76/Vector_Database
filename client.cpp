// =============================================================================
// client.cpp
// Simple CLI client for the Vector Database Engine.
//
// Usage:  ./client <host> <port>
// Example: ./client 127.0.0.1 9000
// =============================================================================

#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

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

static bool send_line(int fd, const std::string& msg) {
    std::string out = msg + "\n";
    ssize_t sent = send(fd, out.c_str(), out.size(), MSG_NOSIGNAL);
    return sent == (ssize_t)out.size();
}

static void recv_response(int fd) {
    std::string status = recv_line(fd);
    if (status.empty()) {
        std::cout << "[disconnected]\n";
        return;
    }
    std::cout << status << "\n";

    int n_results = 0;
    if (status.rfind("OK ", 0) == 0) {
        const char* p = status.c_str() + 3;
        char* end;
        long n = std::strtol(p, &end, 10);
        if (end != p && std::strncmp(end, " results", 8) == 0) {
            n_results = (int)n;
        }
    }

    for (int i = 0; i < n_results; ++i) {
        std::string row = recv_line(fd);
        if (row.empty()) {
            std::cout << "[disconnected mid-response]\n";
            return;
        }
        std::cout << "  " << row << "\n";
    }
}

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