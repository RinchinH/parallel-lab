#include <bits/stdc++.h>
#ifdef _OPENMP
  #include <omp.h>
#else
  // Fallback if OpenMP is not available (approximate wtime)
  #include <chrono>
  double omp_get_wtime() {
      using clock = std::chrono::high_resolution_clock;
      static const auto t0 = clock::now();
      auto now = clock::now();
      std::chrono::duration<double> d = now - t0;
      return d.count();
  }
  void omp_set_num_threads(int) {}
#endif

struct Args {
    int    N = 2000;
    double eps = 1e-8;
    int    maxit = 10000;
    unsigned seed = 42;
    bool   quiet = false;
    bool   dense = false;   // by default use fast path (structure of A)
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
        else if (s=="--dense")                 a.dense= true;
        else if (s=="--fast")                  a.dense= false;
        else {
            // ignore unknown for simplicity
        }
    }
    return a;
}

static inline double l2(const std::vector<double>& v){
    double s=0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int i=0;i<(int)v.size();++i) s += v[i]*v[i];
    return std::sqrt(s);
}

int main(int argc, char** argv){
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Args arg = parse_args(argc, argv);
    const int N = arg.N;

    std::mt19937 rng(arg.seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);

    std::vector<double> x_true(N), b(N), x(N,0.0), x_new(N,0.0);
    for (int i=0;i<N;++i) x_true[i] = U(rng);

    // Build A and b = A x_true
    std::vector<double> A;
    if (arg.dense) {
        A.assign((size_t)N*(size_t)N, 1.0);
        #pragma omp parallel for schedule(static)
        for (int i=0;i<N;++i) A[(size_t)i*N + i] = 2.0*N + 1.0;
        // b = A * x_true (dense)
        #pragma omp parallel for schedule(static)
        for (int i=0;i<N;++i){
            const double* Ai = &A[(size_t)i*N];
            double s = 0.0;
            for (int j=0;j<N;++j) s += Ai[j] * x_true[j];
            b[i] = s;
        }
    } else {
        // Fast path uses structure A = J + 2N*I
        double sum_xt = 0.0;
        for (int j=0;j<N;++j) sum_xt += x_true[j];
        #pragma omp parallel for schedule(static)
        for (int i=0;i<N;++i){
            b[i] = 2.0*N * x_true[i] + sum_xt;
        }
    }

    const double norm_b = l2(b);
    if (!arg.quiet) {
        std::cout << "N="<<N<<", eps="<<arg.eps<<", maxit="<<arg.maxit
                  <<", mode="<<(arg.dense?"dense":"fast")<<"\n";
    }

    // Jacobi iterations
    double t0 = omp_get_wtime();
    int it=0;
    double residual = 1e300;

    while (it < arg.maxit){
        if (arg.dense){
            #pragma omp parallel for schedule(static)
            for (int i=0;i<N;++i){
                const double* Ai = &A[(size_t)i*N];
                double sum = 0.0;
                for (int j=0;j<N;++j){
                    if (j==i) continue;
                    sum += Ai[j]*x[j];
                }
                x_new[i] = (b[i] - sum) / Ai[i];
            }
        } else {
            double sum_x = 0.0;
            #pragma omp parallel for reduction(+:sum_x) schedule(static)
            for (int j=0;j<N;++j) sum_x += x[j];
            #pragma omp parallel for schedule(static)
            for (int i=0;i<N;++i){
                double sum_except_i = sum_x - x[i];
                x_new[i] = (b[i] - sum_except_i) / (2.0*N + 1.0);
            }
        }

        std::swap(x, x_new);

        // Residual r = A x - b
        double r2 = 0.0;
        if (arg.dense){
            #pragma omp parallel for reduction(+:r2) schedule(static)
            for (int i=0;i<N;++i){
                const double* Ai = &A[(size_t)i*N];
                double s = 0.0;
                for (int j=0;j<N;++j) s += Ai[j]*x[j];
                double ri = s - b[i];
                r2 += ri*ri;
            }
        } else {
            double sum_x = 0.0;
            #pragma omp parallel for reduction(+:sum_x) schedule(static)
            for (int j=0;j<N;++j) sum_x += x[j];
            #pragma omp parallel for reduction(+:r2) schedule(static)
            for (int i=0;i<N;++i){
                double s = 2.0*N * x[i] + sum_x;
                double ri = s - b[i];
                r2 += ri*ri;
            }
        }
        residual = std::sqrt(r2) / norm_b;
        ++it;

        if (!arg.quiet && (it % 50 == 0)) {
            std::cout << "it="<<it<<"  residual="<<residual<<"\n";
        }
        if (residual < arg.eps) break;
    }
    double t1 = omp_get_wtime();

    // relative error to original x_true
    std::vector<double> diff(N);
    #pragma omp parallel for schedule(static)
    for (int i=0;i<N;++i) diff[i] = x[i] - x_true[i];
    double rel_err = l2(diff) / l2(x_true);

    if (arg.quiet) {
        std::cout << (t1 - t0) << "," << it << "," << residual << "," << rel_err << "\n";
    } else {
        std::cout << "Finished in " << (t1 - t0) << " s, iters="<<it
                  << ", residual="<<residual
                  << ", rel_err="<<rel_err << "\n";
    }
    return 0;
}
