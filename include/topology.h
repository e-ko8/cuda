#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include "check.h"

#include <string>
#include <vector>

using namespace std;

struct RankHostInfo
{
        int rank;
        char name[HOST_NAME_MAX];
};

// Discover the node/process topology of the current MPI program
void discoverTopology(std::vector<RankHostInfo>& rhis, std::vector<std::string>& hosts, std::vector<int>& hostranks);

#endif // TOPOLOGY_H

