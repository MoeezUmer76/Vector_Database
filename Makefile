# =============================================================================
# Makefile for VecDB (Phase 1 + 2 + 3)
# Usage:
#   make              – build server, client, benchmark
#   make server       – build server only
#   make client       – build client only
#   make benchmark    – build benchmark only
#   make test         – build and run all tests
#   make clean        – remove binaries and object files
# =============================================================================

CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -pthread

# ---- Source lists -----------------------------------------------------------
SERVER_SRCS    := server.cpp vector_store.cpp command_parser.cpp \
                  ivf_index.cpp snapshot.cpp
CLIENT_SRCS    := client.cpp
BENCHMARK_SRCS := benchmark.cpp
TEST2_SRCS     := test_phase2.cpp vector_store.cpp command_parser.cpp ivf_index.cpp
TEST3_SRCS     := test_phase3.cpp vector_store.cpp command_parser.cpp \
                  ivf_index.cpp snapshot.cpp

# ---- Object files -----------------------------------------------------------
SERVER_OBJS    := $(SERVER_SRCS:.cpp=.o)
CLIENT_OBJS    := $(CLIENT_SRCS:.cpp=.o)
BENCHMARK_OBJS := $(BENCHMARK_SRCS:.cpp=.o)

.PHONY: all server client benchmark test clean

all: server client benchmark

server: $(SERVER_OBJS)
	$(CXX) $(CXXFLAGS) -o server $(SERVER_OBJS)
	@echo "Built: ./server"

client: $(CLIENT_OBJS)
	$(CXX) $(CXXFLAGS) -o client $(CLIENT_OBJS)
	@echo "Built: ./client"

benchmark: $(BENCHMARK_OBJS)
	$(CXX) $(CXXFLAGS) -o benchmark $(BENCHMARK_OBJS)
	@echo "Built: ./benchmark"

test: test_phase2 test_phase3
	@echo "\n--- Running Phase 2 tests ---"
	./test_phase2
	@echo "\n--- Running Phase 3 tests ---"
	./test_phase3

test_phase2: $(TEST2_SRCS)
	$(CXX) $(CXXFLAGS) -o test_phase2 $(TEST2_SRCS)

test_phase3: $(TEST3_SRCS)
	$(CXX) $(CXXFLAGS) -o test_phase3 $(TEST3_SRCS)

# Generic rule: .cpp → .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f server client benchmark test_phase2 test_phase3
	rm -f *.o *.vdb *.vdb.tmp
	@echo "Cleaned."
