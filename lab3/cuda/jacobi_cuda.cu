#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iostream>

#define CUDA_CHECK(call) do {                                      \
  cudaError_t err = (call);                                        \
  if (err != cudaSuccess) {                                        \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
            __FILE__, __LINE__, cudaGetErrorString(err));          \
    std::exit(1);                                                  \
  }                                                                \
} while(0)

struct Args {
    int    N = 1000000;
    double eps = 1e-8;
    int    maxit = 10000;
    unsigned seed = 42;
    bool   quiet = false;
    int    block = 256;              // threads per block
    std::string res = "atomic";      // atomic | host
    bool   device_info = false;
};

Args parse_args(int argc, char** argv){
    Args a;
    for (int i=1;i<argc;i++){
        std::string s = argv[i];
        auto next = [&](const char* name){
            if (i+1 >= argc) { throw std::runtime_error(std::string("argument expected for ")+name); }
            return std::string(argv[++i]);
        };
        if (s=="-n" || s=="--size")            a.N    = std::stoi(next("--size"));
        else if (s=="-eps" || s=="--eps")      a.eps  = std::stod(next("--eps"));
        else if (s=="-k" || s=="--maxit")      a.maxit= std::stoi(next("--maxit"));
        else if (s=="-s" || s=="--seed")       a.seed = (unsigned)std::stoul(next("--seed"));
        else if (s=="-q" || s=="--quiet")      a.quiet= true;
        else if (s=="--block")                 a.block= std::stoi(next("--block"));
        else if (s=="--res")                   a.res  = next("--res");   // atomic|host
        else if (s=="--device-info")           a.device_info = true;
    }
    return a;
}

static void print_device_info(){
    int dev = 0;
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    std::cout << "GPU: " << p.name << "\n";
    std::cout << "  Compute Capability: " << p.major << "." << p.minor << "\n";
    std::cout << "  SM count: " << p.multiProcessorCount << "\n";
    std::cout << "  Global memory: " << (double)p.totalGlobalMem / (1024.0*1024.0*1024.0) << " GB\n";
    std::cout << "  Shared mem / block: " << p.sharedMemPerBlock << " bytes\n";
    std::cout << "  Warp size: " << p.warpSize << "\n";
    std::cout << "  Max threads / block: " << p.maxThreadsPerBlock << "\n";
    double mem_bw = 2.0 * (double)p.memoryClockRate * 1000.0 * ((double)p.memoryBusWidth/8.0) / 1e9;
    std::cout << "  Peak mem BW (rough): " << mem_bw << " GB/s\n";
}

// ---------------- Reduction for sum(x) on GPU (shared memory) ----------------
__global__ void reduce_sum_kernel(const double* __restrict__ in,
                                  double* __restrict__ out,
                                  int n)
{
    extern __shared__ double s[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * (blockDim.x * 2) + tid;

    double x = 0.0;
    if (i < (unsigned)n) x += in[i];
    if (i + blockDim.x < (unsigned)n) x += in[i + blockDim.x];

    s[tid] = x;
    __syncthreads();

    for (unsigned s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) s[tid] += s[tid + s2];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

double device_sum(const double* d_in, int n, int block,
                  double* d_buf1, double* d_buf2)
{
    if (n <= 0) return 0.0;
    const double* src = d_in;
    double* dst = d_buf1;

    int cur = n;
    while (cur > 1) {
        int threads = block;
        int blocks = (cur + threads*2 - 1) / (threads*2);
        size_t shmem = (size_t)threads * sizeof(double);
        reduce_sum_kernel<<<blocks, threads, shmem>>>(src, dst, cur);
        CUDA_CHECK(cudaGetLastError());

        cur = blocks;
        src = dst;
        dst = (dst == d_buf1) ? d_buf2 : d_buf1;
    }

    double h = 0.0;
    CUDA_CHECK(cudaMemcpy(&h, src, sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}

// ---------------- Jacobi fast update (A = J + 2N*I) ----------------
__global__ void jacobi_fast_kernel(const double* __restrict__ x,
                                   double* __restrict__ x_new,
                                   const double* __restrict__ b,
                                   int n, double sum_x_old)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    double sum_except = sum_x_old - x[i];
    double denom = (2.0 * n + 1.0);
    x_new[i] = (b[i] - sum_except) / denom;
}

// ---------------- Residual r = Ax - b, atomicAdd accumulation ----------------
// NOTE: atomicAdd on double requires CC >= 6.0 (OK for V100/A100).
__global__ void residual_r2_atomic_kernel(const double* __restrict__ x,
                                         const double* __restrict__ b,
                                         int n, double sum_x,
                                         double* __restrict__ r2_out)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;

    double ri = (2.0 * n) * x[i] + sum_x - b[i];
    atomicAdd(r2_out, ri * ri);
}

// ---------------- Host helpers ----------------
static inline double l2_host(const std::vector<double>& v){
    long double s = 0.0L;
    for (double x : v) s += (long double)x * (long double)x;
    return std::sqrt((double)s);
}

int main(int argc, char** argv){
    Args arg = parse_args(argc, argv);

    if (arg.device_info) {
        print_device_info();
    }

    const int N = arg.N;
    if (arg.block != 64 && arg.block != 128 && arg.block != 256 && arg.block != 512 && arg.block != 1024) { 
        std::cerr << "Use --block 64|128|256|512|1024\n";
        return 1;
    }
    if (arg.res != "atomic" && arg.res != "host") {
        std::cerr << "Use --res atomic|host\n";
        return 1;
    }

    // ---------- Build x_true and b on host ----------
    std::mt19937 rng(arg.seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    std::vector<double> x_true(N), b(N), x_host(N, 0.0), x_new_host(N, 0.0);
    for (int i=0;i<N;++i) x_true[i] = U(rng);

    long double sum_xt = 0.0L;
    for (int i=0;i<N;++i) sum_xt += (long double)x_true[i];

    for (int i=0;i<N;++i) {
        // b_i = 2N*x_i + sum(x_true)
        b[i] = (2.0 * N) * x_true[i] + (double)sum_xt;
    }

    const double norm_b = l2_host(b);

    // ---------- Device alloc ----------
    double *d_x=nullptr, *d_xnew=nullptr, *d_b=nullptr;
    CUDA_CHECK(cudaMalloc(&d_x,    (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_xnew, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b,    (size_t)N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_b, b.data(), (size_t)N*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, (size_t)N*sizeof(double)));

    // Temp buffers for reduction
    int max_blocks = (N + arg.block*2 - 1) / (arg.block*2);
    if (max_blocks < 1) max_blocks = 1;
    double *d_buf1=nullptr, *d_buf2=nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf1, (size_t)max_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_buf2, (size_t)max_blocks * sizeof(double)));

    // Scalar for atomic residual sum
    double* d_r2=nullptr;
    CUDA_CHECK(cudaMalloc(&d_r2, sizeof(double)));

    int grid = (N + arg.block - 1) / arg.block;

    if (!arg.quiet) {
        std::cout << "N="<<N<<", eps="<<arg.eps<<", maxit="<<arg.maxit
                  <<", res="<<arg.res<<", block="<<arg.block<<"\n";
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    int it = 0;
    double residual = 1e300;

    if (arg.res == "atomic") {
        // GPU-heavy mode: residual uses atomicAdd on GPU.
        while (it < arg.maxit) {
            double sum_x = device_sum(d_x, N, arg.block, d_buf1, d_buf2);

            jacobi_fast_kernel<<<grid, arg.block>>>(d_x, d_xnew, d_b, N, sum_x);
            CUDA_CHECK(cudaGetLastError());

            std::swap(d_x, d_xnew);

            double sum_x_new = device_sum(d_x, N, arg.block, d_buf1, d_buf2);

            CUDA_CHECK(cudaMemset(d_r2, 0, sizeof(double)));
            residual_r2_atomic_kernel<<<grid, arg.block>>>(d_x, d_b, N, sum_x_new, d_r2);
            CUDA_CHECK(cudaGetLastError());

            double r2 = 0.0;
            CUDA_CHECK(cudaMemcpy(&r2, d_r2, sizeof(double), cudaMemcpyDeviceToHost));
            residual = std::sqrt(r2) / norm_b;

            ++it;
            if (!arg.quiet && (it % 50 == 0)) {
                std::cout << "it="<<it<<"  residual="<<residual<<"\n";
            }
            if (residual < arg.eps) break;
        }
    } else {
        // Host-check mode: copy x_new to host each iter, compute residual on CPU.
        while (it < arg.maxit) {
            long double sum_x_old = 0.0L;
            for (int i=0;i<N;++i) sum_x_old += (long double)x_host[i];

            CUDA_CHECK(cudaMemcpy(d_x, x_host.data(), (size_t)N*sizeof(double), cudaMemcpyHostToDevice));

            jacobi_fast_kernel<<<grid, arg.block>>>(d_x, d_xnew, d_b, N, (double)sum_x_old);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(x_new_host.data(), d_xnew, (size_t)N*sizeof(double), cudaMemcpyDeviceToHost));

            long double sum_x_new = 0.0L;
            for (int i=0;i<N;++i) sum_x_new += (long double)x_new_host[i];

            long double r2 = 0.0L;
            for (int i=0;i<N;++i) {
                long double ri = (long double)(2.0*N) * (long double)x_new_host[i] + sum_x_new - (long double)b[i];
                r2 += ri*ri;
            }
            residual = std::sqrt((double)r2) / norm_b;

            x_host.swap(x_new_host);
            ++it;

            if (!arg.quiet && (it % 50 == 0)) {
                std::cout << "it="<<it<<"  residual="<<residual<<"\n";
            }
            if (residual < arg.eps) break;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_sec = std::chrono::duration<double>(t1 - t0).count();

    // rel_err to x_true
    std::vector<double> x_final(N);
    if (arg.res == "atomic") {
        CUDA_CHECK(cudaMemcpy(x_final.data(), d_x, (size_t)N*sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        x_final = x_host;
    }
    std::vector<double> diff(N);
    for (int i=0;i<N;++i) diff[i] = x_final[i] - x_true[i];
    double rel_err = l2_host(diff) / l2_host(x_true);

    if (arg.quiet) {
        std::cout << time_sec << "," << it << "," << residual << "," << rel_err << "\n";
    } else {
        std::cout << "Finished in " << time_sec << " s, iters="<<it
                  << ", residual="<<residual
                  << ", rel_err="<<rel_err << "\n";
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_xnew));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_buf1));
    CUDA_CHECK(cudaFree(d_buf2));
    CUDA_CHECK(cudaFree(d_r2));
    return 0;
}
