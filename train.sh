#!/usr/bin/env sh

TOOLS=./build/tools

#MPI=/usr/local/mvapich2-2.2rc1/bin
#MPI=/usr/local/openmpi-1.8.5/bin
#MPI=/home/lixiangqiao/mpich/bin
#$MPI/mpirun_rsh -hostfile ./hostfile -np 2  MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
        $TOOLS/caffe train \
  --solver=models/bvlc_googlenet/solver.prototxt

