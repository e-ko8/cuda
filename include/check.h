#ifndef CHECK_H
#define CHECK_H

#include <chrono>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

#ifndef HOST_NAME_MAX
# if defined(_POSIX_HOST_NAME_MAX)
#  define HOST_NAME_MAX _POSIX_HOST_NAME_MAX
# elif defined(MAXHOSTNAMELEN)
#  define HOST_NAME_MAX MAXHOSTNAMELEN
# endif
#endif /* HOST_NAME_MAX */

#define CUDA_ERR_CHECK(x)                                         \
    do { cudaError_t err = x; if (err != cudaSuccess) {           \
        char hostname[HOST_NAME_MAX] = "";                        \
        gethostname(hostname, HOST_NAME_MAX);                     \
        fprintf(stderr, "CUDA error %d \"%s\" on %s at %s:%d\n",  \
            (int)err, cudaGetErrorString(err), hostname,          \
            __FILE__, __LINE__);                                  \
        if (!getenv("FREEZE_ON_ERROR")) {                         \
            fprintf(stderr, "You may want to set "                \
                "FREEZE_ON_ERROR environment "                    \
                "variable to debug the case\n");                  \
            exit(-1);                                             \
        }                                                         \
        else {                                                    \
            fprintf(stderr, "thread 0x%zx of pid %d @ %s "        \
                "is entering infinite loop\n",                    \
                (size_t)pthread_self(), (int)getpid(), hostname); \
            while (1) std::this_thread::sleep_for(                \
                std::chrono::seconds(1)); /* 1 sec */             \
        }                                                         \
    }} while (0)

#define MPI_ERR_CHECK(call)                                       \
    do { int err = call;                                          \
    if (err != MPI_SUCCESS) {                                     \
        char hostname[HOST_NAME_MAX] = "";                        \
        gethostname(hostname, HOST_NAME_MAX);                     \
        char errstr[MPI_MAX_ERROR_STRING];                        \
        int szerrstr;                                             \
        MPI_Error_string(err, errstr, &szerrstr);                 \
        fprintf(stderr, "MPI error on %s at %s:%i : %s\n",        \
            hostname, __FILE__, __LINE__, errstr);                \
        if (!getenv("FREEZE_ON_ERROR")) {                         \
            fprintf(stderr, "You may want to set "                \
                "FREEZE_ON_ERROR environment "                    \
                "variable to debug the case\n");                  \
            MPI_Abort(MPI_COMM_WORLD, err);                       \
        }                                                         \
        else {                                                    \
            fprintf(stderr, "thread 0x%zx of pid %d @ %s "        \
                "is entering infinite loop\n",                    \
                (size_t)pthread_self(), (int)getpid(), hostname); \
            while (1) std::this_thread::sleep_for(                \
                std::chrono::seconds(1)); /* 1 sec */             \
        }                                                         \
    }} while (0)

#endif // CHECK_H

