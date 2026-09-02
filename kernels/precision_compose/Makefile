# Standalone icpx AOT for BMG-G31. Source scripts/oneapi-env.sh first,
# or use ./compile_in_docker.sh (CPU docker, no --device).
CXX ?= icpx
TARGET ?= intel_gpu_bmg_g31
CXXFLAGS ?= -fsycl -fsycl-targets=$(TARGET) -O3 -std=c++17 -Wall
BIN_DIR ?= bin

SRCS = compose_s8_from_s4.cpp compose_s8_from_s4_karatsuba.cpp compose_e2m1_two_term.cpp
BINS = $(SRCS:%.cpp=$(BIN_DIR)/%)

.PHONY: all clean
all: $(BINS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: %.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
