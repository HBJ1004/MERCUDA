# Ordinary builds/runs still use make and ./mercury6; settings live in .in files.
.DEFAULT_GOAL := build
FC = gfortran
CXX ?= g++
FFLAGS ?= -O3 -g -ffixed-line-length-none -ffp-contract=off
CXXFLAGS ?= -O3 -g -std=c++17 -ffp-contract=off
NVCC ?= $(shell command -v nvcc 2>/dev/null)
ifeq ($(strip $(NVCC)),)
NVCC := $(firstword $(wildcard /usr/local/cuda/bin/nvcc $(HOME)/.local/opt/cuda-*/bin/nvcc))
endif
CUDA_ARCH ?= native
CUDA ?= auto
BUILD ?= build
BIN ?= .
EXEC ?= .
ifeq ($(CUDA),0)
override NVCC :=
endif
ifneq ($(strip $(NVCC)),)
NVCC_PATH := $(shell command -v $(NVCC))
CUDA_ROOT := $(abspath $(dir $(NVCC_PATH))/..)
CUDA_LIB := $(firstword $(wildcard $(CUDA_ROOT)/lib64/libcudart_static.a $(CUDA_ROOT)/lib/libcudart_static.a $(CUDA_ROOT)/targets/x86_64-linux/lib/libcudart_static.a /usr/lib/x86_64-linux-gnu/libcudart_static.a))
ifeq ($(strip $(CUDA_LIB)),)
$(error CUDA runtime library not found; set CUDA_LIB=/path/to/libcudart_static.a or CUDA=0)
endif
GPU_OBJECT := $(BUILD)/mercury_cuda.o
GPU_TEST := 1
GPU_LIBS := $(CUDA_LIB) -ldl -lrt -pthread
else
GPU_OBJECT := $(BUILD)/mercury_cuda_stub.o
GPU_LIBS :=
GPU_TEST := 0
endif
SUPPORT := $(BUILD)/mercury_support.o
GPU_SUPPORT := $(BUILD)/mercury_gpu.o
FDEPENDS := mercury.inc swift.inc
.PHONY: build cpu test test-debug clean clean-build unbuild help gen-in rm-gen rm-in FORCE
build: $(BIN)/mercury6 $(BIN)/element6 $(BIN)/close6
$(BUILD)/.dir:
	mkdir -p $(BUILD)
	touch $@
$(SUPPORT): mercury_support.f90 | $(BUILD)/.dir
	$(FC) $(FFLAGS) -J$(BUILD) -I$(BUILD) -c $< -o $@
$(GPU_SUPPORT): mercury_gpu.f90 $(SUPPORT)
	$(FC) $(FFLAGS) -J$(BUILD) -I$(BUILD) -c $< -o $@
$(BUILD)/mercury_cuda.o: mercury_cuda.cu mercury_cuda.h mercury_cuda_adaptive.cuh mercury_cuda_radau.cuh | $(BUILD)/.dir
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) --fmad=false -Xcompiler -fPIC -c $< -o $@
$(BUILD)/mercury_cuda_stub.o: mercury_cuda_stub.cpp mercury_cuda.h | $(BUILD)/.dir
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(BUILD)/mercury6.o: mercury6_2.for $(FDEPENDS) $(GPU_SUPPORT)
	$(FC) $(FFLAGS) -I$(BUILD) -c $< -o $@
$(BIN)/mercury6: $(BUILD)/mercury6.o $(SUPPORT) $(GPU_SUPPORT) $(GPU_OBJECT) FORCE
	mkdir -p $(dir $@)
	$(FC) $(FFLAGS) -o $@ $(filter %.o,$^) $(GPU_LIBS) -lstdc++
$(BIN)/element6: element6.for $(FDEPENDS) $(SUPPORT)
	mkdir -p $(dir $@)
	$(FC) $(FFLAGS) -I$(BUILD) -o $@ $< $(SUPPORT)
$(BIN)/close6: close6.for $(FDEPENDS) $(SUPPORT)
	mkdir -p $(dir $@)
	$(FC) $(FFLAGS) -I$(BUILD) -o $@ $< $(SUPPORT)
cpu:
	$(MAKE) CUDA=0 build
test: build
	MERCURY_TEST_BIN=$(abspath $(BIN)) MERCURY_TEST_BUILD=$(abspath $(BUILD)) MERCURY_TEST_CUDA=$(GPU_TEST) python3 -m unittest discover -s tests -p 'test_*.py' -v
test-debug:
	$(MAKE) BUILD=build/debug BIN=build/debug FFLAGS='-O0 -g -ffixed-line-length-none -ffp-contract=off -fcheck=all -fbacktrace' test
help:
	@echo 'make [build] | make cpu | make test | make test-debug'
	@echo 'make clean-build: remove compiled files and executables; keep simulation files'
	@echo 'make rm-gen: remove simulation outputs and dumps; make rm-in: remove inputs'
	@echo 'make clean: remove compiled files, executables, simulation outputs and inputs'
	@echo 'Run ./mercury6, ./element6 or ./close6 with settings in .in files.'
	@echo 'Optional build settings: CUDA=0, NVCC=/path/to/nvcc, CUDA_ARCH=sm_89'
clean: clean-build rm-gen rm-in
	rm -rf $(EXEC)/*.dSYM
clean-build:
	rm -rf $(BUILD)
	rm -f $(BIN)/mercury6 $(BIN)/element6 $(BIN)/close6
unbuild:
	rm -f $(BIN)/mercury6 $(BIN)/element6 $(BIN)/close6
gen-in:
	@for sample in *.in.sample; do \
	  target=$${sample%.sample}; \
	  if test -e "$$target"; then echo "Keeping $$target"; else cp "$$sample" "$$target"; fi; \
	done
rm-gen:
	rm -f $(EXEC)/*.dmp $(EXEC)/*.clo $(EXEC)/*.out $(EXEC)/*.tmp $(EXEC)/*.aei
rm-in:
	rm -f $(EXEC)/*.in
