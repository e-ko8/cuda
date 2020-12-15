#include <mpi.h>
#include "check.h"
#include "topology.h"

using namespace std;

// Discover the node/process topology of the current MPI program 
void discoverTopology(vector<RankHostInfo>& rhis, vector<string>& hosts, vector<int>& hostranks)
{
	int rank, size, root = 0;
	bool master;

	rhis.resize(1);
	{
		// Figure out my host name.
		RankHostInfo rhi;
		if (gethostname(rhi.name, HOST_NAME_MAX) != 0)
		{
			fprintf(stderr, "Error in gethostname: errno = %d\n", errno);
			exit(-1);
		}

		MPI_ERR_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
		MPI_ERR_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
		master = !rank;
	
		// Collect host names from all nodes.
		rhi.rank = rank;
		if (master) rhis.resize(size);
		MPI_ERR_CHECK(MPI_Gather(
			&rhi, sizeof(RankHostInfo), MPI_BYTE, &rhis[0],
			sizeof(RankHostInfo), MPI_BYTE, root, MPI_COMM_WORLD));
	}
	
	// Find groups on master only and send them to all other nodes.
	// Gather nodes belonging to the same host name into MPI_COMM_WORLDunicators.
	int nhosts;
	if (master)
	{
		// Track already discovered groups of ranks with the same
		// hostname.
		int ngroups = 0;
		vector<string> groups;
		groups.resize(size);
		for (int rank = 0; rank < size; rank++)
		{
			string hostname = rhis[rank].name;
			
			// Do not process again previously seen hostname.
			bool seen = false;
			for (int i = 0; i < ngroups; i++)
				if (hostname == groups[i])
				{
					seen = true;
					break;
				}
			if (seen) continue;

			groups[ngroups++] = hostname;

			// Count how many ranks are there with the current hostname.
			int nhostranks = 0;
			for (int i = 0, e = rhis.size(); i != e; i++)
				if (hostname == rhis[i].name)
					nhostranks++;
			
			// Record those ranks.
			vector<int> group;
			group.resize(nhostranks);
			nhostranks = 0;
			for (int i = 0, e = rhis.size(); i != e; i++)
				if (hostname == rhis[i].name)
					group[nhostranks++] = rhis[i].rank;
			
			// Send size of group and the ranks themselves to every
			// member of group. Do not send to the master node himself.
			vector<MPI_Request> requests;
			requests.resize(nhostranks * 2);
			for (int i = 0; i < nhostranks; i++)
			{
				int rank = group[i];
				if (rank == root)
				{
					// Fill the hostranks on the master node w/o send/recv.
					hostranks.resize(nhostranks);
					hostranks.assign(group.begin(), group.end());
					continue;
				}
				MPI_ERR_CHECK(MPI_Isend(&nhostranks, 1, MPI_INT, group[i], 0,
					MPI_COMM_WORLD, &requests[2 * i]));
				MPI_ERR_CHECK(MPI_Isend(&group[0], nhostranks, MPI_INT, group[i], 1,
					MPI_COMM_WORLD, &requests[2 * i + 1]));
			}
			for (int i = 0; i < 2 * nhostranks; i++)
			{
				if (rank == 2 * root) continue;
				if (rank == 2 * root + 1) continue;
				MPI_ERR_CHECK(MPI_Wait(&requests[i], MPI_STATUS_IGNORE));
			}
		}
		nhosts = ngroups;
		hosts.resize(nhosts);
		printf("%d host(s) participating: \n", nhosts);
		for (int i = 0; i < nhosts; i++)
		{
			printf("\t%s\n", groups[i].c_str());
			hosts[i] = groups[i];
		}
		printf("\n");
	}
	else
	{
		// Receive my group size & ranks from master.
		int nhostranks = 0;
		MPI_ERR_CHECK(MPI_Recv(&nhostranks, 1, MPI_INT, root, 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE));
		hostranks.resize(nhostranks);
		MPI_ERR_CHECK(MPI_Recv(&hostranks[0], nhostranks, MPI_INT, root, 1,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE));
	}

	// Tell everyone the number of available hosts.
	// Can do MPI_Ibcast here, but it's not available in some implementations.
	vector<MPI_Request> requests;
	if (master)
	{
		requests.resize(size - 1);
		for (int i = 0, ireq = 0; i < size; i++)
		{
			if (i == root) continue;
			MPI_ERR_CHECK(MPI_Isend(&nhosts, 1, MPI_INT, i, 0,
				MPI_COMM_WORLD, &requests[ireq++]));
		}
	}
	else
	{
		requests.resize(1);
		MPI_ERR_CHECK(MPI_Irecv(&nhosts, 1, MPI_INT, root, 0,
			MPI_COMM_WORLD, &requests[0]));
	}
	
	if (hostranks.size() == 0)
	{
		printf("The number of ranks in hostname group cannot be zero,"
			" something went wrong, aborting\n");
		abort();
	}
			
	MPI_ERR_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

