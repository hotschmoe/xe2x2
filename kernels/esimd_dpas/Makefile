# Standalone icpx AOT for BMG-G31. Source scripts/oneapi-env.sh first,
# or use ./compile_in_docker.sh (CPU docker, no --device).
CXX ?= icpx
TARGET ?= intel_gpu_bmg_g31
CXXFLAGS ?= -fsycl -fsycl-targets=$(TARGET) -O3 -std=c++17 -Wall
BIN_DIR ?= bin

SRCS = dpas_s8.cpp dpas_s4.cpp dpas_s2xs8.cpp dpas_s2.cpp
BINS = $(SRCS:%.cpp=$(BIN_DIR)/%)

.PHONY: all clean
all: $(BINS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: %.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
