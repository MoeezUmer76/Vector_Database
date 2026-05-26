#pragma once
// =============================================================================
// command_parser.h
// Parses the line-based text protocol into typed command structs.
// Protocol summary:
//   ADD    <id> <v1> <v2> ... <vD>
//   SEARCH <v1> <v2> ... <vD> <k> BRUTE
//   SEARCH <v1> <v2> ... <vD> <k> IVF <nprobe>
//   BUILD
//   SAVE
//   LOAD
//   STATS
//   QUIT
// =============================================================================

#include <string>
#include <vector>
#include <cstdint>

enum class CommandType {
    ADD,
    SEARCH,
    BUILD,
    SAVE,
    LOAD,
    STATS,
    QUIT,
    UNKNOWN
};

// Search method selected by the parser
enum class SearchMethod { BRUTE, IVF };

struct Command {
    CommandType  type   = CommandType::UNKNOWN;
    uint64_t     id     = 0;           // used by ADD
    std::vector<float> floats;         // used by ADD and SEARCH
    int          k      = 0;           // used by SEARCH
    SearchMethod method = SearchMethod::BRUTE; // used by SEARCH
    int          nprobe = 1;           // used by SEARCH IVF
    std::string  error;                // non-empty when parsing failed
};

// Parse a single trimmed line and return the corresponding Command.
Command parse_command(const std::string& line, int expected_dim);
