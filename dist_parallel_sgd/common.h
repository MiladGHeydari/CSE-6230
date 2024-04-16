#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <cstdint>
#include <mpi.h>

void train(double* dataset, double* w, int rank, int num_procs);

#endif
