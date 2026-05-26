// =============================================================================
// command_parser.cpp
// =============================================================================

#include "command_parser.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Helper: trim leading/trailing whitespace from a string
// ---------------------------------------------------------------------------
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// ---------------------------------------------------------------------------
// Helper: convert string to uppercase in-place
// ---------------------------------------------------------------------------
static std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

// ---------------------------------------------------------------------------
Command parse_command(const std::string& raw_line, int expected_dim) {
    Command cmd;
    std::string line = trim(raw_line);

    if (line.empty()) {
        cmd.error = "ERR empty command";
        return cmd;
    }

    // Tokenise by whitespace
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string tok;
    while (iss >> tok) tokens.push_back(tok);

    const std::string verb = to_upper(tokens[0]);

    // ------------------------------------------------------------------
    // QUIT
    // ------------------------------------------------------------------
    if (verb == "QUIT") {
        cmd.type = CommandType::QUIT;
        return cmd;
    }

    // ------------------------------------------------------------------
    // STATS
    // ------------------------------------------------------------------
    if (verb == "STATS") {
        cmd.type = CommandType::STATS;
        return cmd;
    }

    // ------------------------------------------------------------------
    // BUILD  (no arguments)
    // ------------------------------------------------------------------
    if (verb == "BUILD") {
        if (tokens.size() != 1) {
            cmd.error = "ERR BUILD takes no arguments";
            return cmd;
        }
        cmd.type = CommandType::BUILD;
        return cmd;
    }

    // ------------------------------------------------------------------
    // SAVE  (no arguments)
    // ------------------------------------------------------------------
    if (verb == "SAVE") {
        if (tokens.size() != 1) {
            cmd.error = "ERR SAVE takes no arguments";
            return cmd;
        }
        cmd.type = CommandType::SAVE;
        return cmd;
    }

    // ------------------------------------------------------------------
    // LOAD  (no arguments)
    // ------------------------------------------------------------------
    if (verb == "LOAD") {
        if (tokens.size() != 1) {
            cmd.error = "ERR LOAD takes no arguments";
            return cmd;
        }
        cmd.type = CommandType::LOAD;
        return cmd;
    }

    // ------------------------------------------------------------------
    // ADD <id> <v1> ... <vD>
    //   tokens: [ADD, id, v1, v2, ..., vD]  → 2 + D tokens total
    // ------------------------------------------------------------------
    if (verb == "ADD") {
        int expected_tokens = 2 + expected_dim;
        if ((int)tokens.size() != expected_tokens) {
            cmd.error = "ERR ADD expects " + std::to_string(expected_tokens)
                      + " tokens (ADD id v1..vD), got "
                      + std::to_string(tokens.size());
            return cmd;
        }
        try {
            cmd.id = std::stoull(tokens[1]);
        } catch (...) {
            cmd.error = "ERR invalid id (must be non-negative integer)";
            return cmd;
        }
        cmd.floats.reserve(expected_dim);
        for (int i = 0; i < expected_dim; ++i) {
            try {
                cmd.floats.push_back(std::stof(tokens[2 + i]));
            } catch (...) {
                cmd.error = "ERR invalid float at position " + std::to_string(i + 1);
                return cmd;
            }
        }
        cmd.type = CommandType::ADD;
        return cmd;
    }

    // ------------------------------------------------------------------
    // SEARCH <v1> ... <vD> <k> BRUTE
    //   tokens: [SEARCH, v1..vD, k, BRUTE]         → 3 + D tokens
    //
    // SEARCH <v1> ... <vD> <k> IVF <nprobe>
    //   tokens: [SEARCH, v1..vD, k, IVF, nprobe]   → 4 + D tokens
    // ------------------------------------------------------------------
    if (verb == "SEARCH") {
        // Need at least: SEARCH + D floats + k + method = D+3 tokens
        if ((int)tokens.size() < 3 + expected_dim) {
            cmd.error = "ERR SEARCH: too few tokens";
            return cmd;
        }

        // Parse D floats
        cmd.floats.reserve(expected_dim);
        for (int i = 0; i < expected_dim; ++i) {
            try {
                cmd.floats.push_back(std::stof(tokens[1 + i]));
            } catch (...) {
                cmd.error = "ERR invalid float at position " + std::to_string(i + 1);
                return cmd;
            }
        }

        // Parse k
        try {
            int k_val = std::stoi(tokens[1 + expected_dim]);
            if (k_val <= 0) throw std::invalid_argument("k must be > 0");
            cmd.k = k_val;
        } catch (...) {
            cmd.error = "ERR invalid k (must be positive integer)";
            return cmd;
        }

        // Parse method keyword
        std::string method_str = to_upper(tokens[2 + expected_dim]);

        if (method_str == "BRUTE") {
            // Expect exactly 3+D tokens
            if ((int)tokens.size() != 3 + expected_dim) {
                cmd.error = "ERR SEARCH BRUTE: unexpected extra tokens";
                return cmd;
            }
            cmd.method = SearchMethod::BRUTE;
            cmd.type   = CommandType::SEARCH;
            return cmd;
        }

        if (method_str == "IVF") {
            // Expect exactly 4+D tokens (extra: nprobe)
            if ((int)tokens.size() != 4 + expected_dim) {
                cmd.error = "ERR SEARCH IVF expects SEARCH v1..vD k IVF nprobe";
                return cmd;
            }
            try {
                int np = std::stoi(tokens[3 + expected_dim]);
                if (np <= 0) throw std::invalid_argument("nprobe must be > 0");
                cmd.nprobe = np;
            } catch (...) {
                cmd.error = "ERR invalid nprobe (must be positive integer)";
                return cmd;
            }
            cmd.method = SearchMethod::IVF;
            cmd.type   = CommandType::SEARCH;
            return cmd;
        }

        cmd.error = "ERR unknown search method '" + method_str + "' (use BRUTE or IVF)";
        return cmd;
    }

    // ------------------------------------------------------------------
    // Unknown verb
    // ------------------------------------------------------------------
    cmd.error = "ERR unknown command '" + verb + "'";
    return cmd;
}
