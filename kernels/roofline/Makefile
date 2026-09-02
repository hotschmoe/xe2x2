# Standalone icpx AOT for BMG-G31. Source scripts/oneapi-env.sh first.
CXX ?= icpx
TARGET ?= intel_gpu_bmg_g31
CXXFLAGS ?= -fsycl -fsycl-targets=$(TARGET) -O3 -std=c++17 -Wall
BIN_DIR ?= .

.PHONY: all clean
all: $(BIN_DIR)/copy_roof $(BIN_DIR)/s8_square_gemm $(BIN_DIR)/s8_gemv

$(BIN_DIR)/copy_roof: copy_roof.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/s8_square_gemm: s8_square_gemm.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN_DIR)/s8_gemv: s8_gemv.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(BIN_DIR)/copy_roof $(BIN_DIR)/s8_square_gemm $(BIN_DIR)/s8_gemv
