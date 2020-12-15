#include <mpi.h>
#include "check.h"
#include "topology.h"

#include <chrono>
#include <thread>

using namespace std;

int main(int argc, char* argv[])
{
	MPI_ERR_CHECK(MPI_Init(&argc, &argv));

	int rank, size;
	MPI_ERR_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	MPI_ERR_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

	// All MPI routines (except MPI_Wtime and MPI_Wtick) return an error value.
	// Before the value is returned, the current MPI error handler is called.
	// By default, this error handler aborts the MPI job. The error handler may be changed
	// with MPI_Comm_set_errhandler (for MPI_COMM_WORLDunicators), MPI_File_set_errhandler (for files),
	// and MPI_Win_set_errhandler (for RMA windows). The predefined error handler
	// MPI_ERRORS_RETURN may be used to cause error values to be returned.
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	vector<RankHostInfo> rhis;
	vector<string> hosts;
	vector<int> hostranks;
	discoverTopology(rhis, hosts, hostranks);

	char hostname[HOST_NAME_MAX];
	gethostname(hostname, HOST_NAME_MAX);

	printf("rank %d (pid = %u) @ hostname \"%s\"", rank, getpid(), hostname);
	if (hostranks.size() - 1)
	{
		printf(", together with %zu other ranks:", hostranks.size() - 1);
		for (int i = 0, e = hostranks.size(); i != e; i++)
			if (rank != hostranks[i])
				printf(" %d", hostranks[i]);
	}
	printf("\n");

	size_t szbuffer = 16 * 1024 * 1024;

	printf("Testing point-to-point performance, message size = %zu bytes:\n", szbuffer);

	vector<vector<double> > buffers(size, vector<double>(szbuffer));
	vector<MPI_Request> requests(size);

	// Set to receive a message from everyone, except myself.
	for (int i = 0; i < size; i++)
	{
		if (rank == i) continue;

		MPI_ERR_CHECK(MPI_Irecv(&buffers[i][0], buffers[i].size(), MPI_DOUBLE, i, 0,
			MPI_COMM_WORLD, &requests[i]));
#if 0
		printf("Set to receive on %d from %d\n", rank, i);
#endif
	}

	// Ensure only one message is passed at a time.
	for (int i = 0; i < size; i++)
	{
		if (rank == i)
		{
			// Send a message to everyone.
			for (int j = 0; j < size; j++)
			{
				if (rank == j) continue;

				// Wait for 0.1 second to ensure the receiving part
				// has arrived earlier and is already waiting.
				std::this_thread::sleep_for(std::chrono::milliseconds(100));

				double start = MPI_Wtime();
#if 0
				printf("Sending from %d to %d\n", rank, j);
#endif
				MPI_ERR_CHECK(MPI_Send(&buffers[rank][0], buffers[rank].size(), MPI_DOUBLE, j, 0, MPI_COMM_WORLD));
				double finish = MPI_Wtime();

				bool hostlocal = false;
				if (hostranks.size() - 1)
				{
					for (int k = 0, e = hostranks.size(); k != e; k++)
					{
						if (hostranks[k] != j) continue;

						hostlocal = true;
						break;
					}
				}

				printf("[%s] %d -> %d time = %f sec (%f MB/sec)\n",
					hostlocal ? "INTRANODE" : "INTERNODE", rank, j, finish - start,
					szbuffer / (finish - start) / 1024 / 1024);
				fflush(stdout);
			}
		}
		else
		{
			// Everybody else waits to receive.
#if 0
			printf("Waiting on %d to receive from %d\n", rank, i);
#endif
			MPI_ERR_CHECK(MPI_Wait(&requests[i], MPI_STATUS_IGNORE));
		}

		MPI_ERR_CHECK(MPI_Barrier(MPI_COMM_WORLD));
	}

#if 0
	for (int i = 0; i < size; i++)
	{
		if (rank == i) continue;
		MPI_ERR_CHECK(MPI_Request_free(&requests[i]));
	}
#endif

	MPI_ERR_CHECK(MPI_Finalize());

	return 0;
}

