#include <mpi.h>
#include "check.h"

#if VERSION == 2
#include "xsimd/xsimd.hpp"
#include <omp.h>

namespace xs = xsimd;

template<size_t T>
void vec_add(xs::batch<double, T> &inout, xs::batch<double, T> &in)
{
	inout += in;
}

#endif

#include <math.h>

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		printf("Usage: %s <n>\n", argv[0]);
		return 0;
	}

	unsigned long n = strtoul(argv[1], NULL, 0);
	printf("n = %lu\n", n);

	volatile double start = MPI_Wtime();

#ifdef AMDAHL
	volatile double startInit = MPI_Wtime();
#endif
	MPI_ERR_CHECK(MPI_Init(&argc, &argv));
#ifdef AMDAHL
	volatile double finishInit = MPI_Wtime();
#endif

	int size, rank;
	MPI_ERR_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
	MPI_ERR_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

#if VERSION == 0
	double h = 1.0 / n;

	// Integrate
	double sum = 0.0;
	for (unsigned long i = rank; i < n; i += size)
	{
		const double x = (i + 0.5) * h;
	        sum += 4.0 / (1.0 + x * x);
	}

	// Aggregate partial sums
	double pi;
	MPI_ERR_CHECK(MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
#elif VERSION == 1
	double h = 1.0 / n;

	// Integrate
	double sum = 0.0;
	#pragma omp parallel for reduction(+:sum)
	for (unsigned long i = rank; i < n; i += size)
	{
		const double x = (i + 0.5) * h;
		sum += 4.0 / (1.0 + x * x);
	}

	// Aggregate partial sums
	double pi;
	MPI_ERR_CHECK(MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
#elif VERSION == 2
        double h = 1.0 / n;

        // Prepare the vector boilerplate.
	constexpr std::size_t simd_size = xs::simd_type<double>::size;
	xs::batch<double, simd_size> vi;
	for (int i = 0; i < simd_size; i++)
		vi[i] = i + 0.5;

	// Integrate
	xs::batch<double, simd_size> sum(0.0);
	#pragma omp declare reduction(vec_add : \
		xs::batch<double, simd_size> : \
                vec_add(omp_out, omp_in) \
	) initializer (omp_priv=omp_orig)
	#pragma omp parallel for reduction(vec_add : sum)
        for (unsigned long i = rank; i < n; i += size * simd_size)
        {
		const auto x = (vi + i) * h;
                sum += 4.0 / (1.0 + x * x);
        }

	// Horizontal sum
	for (int i = 1; i < simd_size; i++)
		sum[0] += sum[i];
#ifdef AMDAHL
        volatile double startReduction = MPI_Wtime();
#endif
	// Aggregate partial sums
        double pi;
        MPI_ERR_CHECK(MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

#ifdef AMDAHL
        volatile double endReduction = MPI_Wtime();
#endif
#endif

	MPI_ERR_CHECK(MPI_Finalize());

	double finish = MPI_Wtime();

	if (rank == 0)
	{
		pi *= h;
		printf("pi = %f, err = %e\n", pi, abs(pi - M_PI));
		printf("time = %f sec\n", finish - start);
#ifdef AMDAHL
		printf("Init time = %f sec\n", finishInit - startInit);
		printf("Reduction time = %f sec\n", endReduction - startReduction);
#endif
	}

	return 0;
}

