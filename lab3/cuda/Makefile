NVCC      = nvcc
TARGET    = jacobi_cuda

# Targets for common cluster GPUs (V100 sm_70, A100 sm_80) + PTX fallback
NVCCFLAGS = -O3 -std=c++17 -lineinfo \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_80,code=compute_80

all: $(TARGET)

$(TARGET): jacobi_cuda.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
