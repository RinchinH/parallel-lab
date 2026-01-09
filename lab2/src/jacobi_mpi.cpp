#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>

double l2_norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (const auto& val : v) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 2000;  
    double eps = 1e-8;  
    int maxit = 10000;  


    std::vector<double> A(N * N, 1.0);
    std::vector<double> b(N, N + 1);  
    std::vector<double> x(N, 0.0);    
    std::vector<double> x_new(N, 0.0); 

    std::ofstream result_file;
    if (rank == 0) {
        result_file.open("output.csv", std::ios::out | std::ios::trunc);
        if (result_file.is_open()) {
            result_file << "threads,N,mode,eps,maxit,time_sec,iters,residual,rel_err\n";
        }
    }

    std::vector<double> times, residuals, rel_errors;
    std::vector<int> iterations;

    for (int np = 1; np <= 16; np *= 2) {  
        if (rank == 0) {
            std::cout << "Running with " << np << " processes..." << std::endl;
        }

        int local_size = N / np;  

        std::vector<double> local_b(local_size);
        std::vector<double> local_x(local_size, 0.0);

        MPI_Scatter(b.data(), local_size, MPI_DOUBLE, local_b.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double t0 = MPI_Wtime();

        int it = 0;
        double residual = 1e300;
        double rel_err = 0.0;

        while (it < maxit) {
            for (int i = 0; i < local_size; ++i) {
                double sum = 0.0;
                for (int j = 0; j < N; ++j) {
                    if (i != j) sum += A[i * N + j] * x[j];
                }
                local_x[i] = (local_b[i] - sum) / (2 * N + 1);  
            }

            MPI_Gather(local_x.data(), local_size, MPI_DOUBLE, x.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            residual = l2_norm(x);
            if (residual < eps) break;

            std::swap(x, local_x);
            ++it;
        }

        double t1 = MPI_Wtime();

        std::vector<double> diff(N);
        for (int i = 0; i < N; ++i) {
            diff[i] = x[i] - b[i];  
        }
        rel_err = l2_norm(diff) / l2_norm(b);

        if (rank == 0) {
            times.push_back(t1 - t0);
            residuals.push_back(residual);
            rel_errors.push_back(rel_err);
            iterations.push_back(it);
        }
    }

    if (rank == 0) {
        for (int i = 0; i < 5; ++i) {
            result_file << (1 << i) << "," << N << ",fast," << eps << "," << maxit
                        << "," << times[i] << "," << iterations[i] << "," << residuals[i] << "," << rel_errors[i] << "\n";
        }
    }

    if (rank == 0) {
        result_file.close();
    }

    MPI_Finalize();
    return 0;
}
