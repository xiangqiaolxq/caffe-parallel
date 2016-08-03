#!/usr/bin/env sh

TOOLS=./build/tools

MPI=/usr/local/mvapich2-2.2rc1/bin
#MPI=/usr/local/openmpi-1.8.5/bin
#MPI=/home/lixiangqiao/mpich/bin
$MPI/mpirun_rsh -hostfile ./hostfile -np 4\
        MV2_USE_CUDA=0 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0  \
        $TOOLS/caffe train \
   --solver=models/googlenet_v3/solver.prototxt  \
#   --snapshot=/data1/lixiangqiao/models/resnet/resnet_sync_iter_220000.solverstate
