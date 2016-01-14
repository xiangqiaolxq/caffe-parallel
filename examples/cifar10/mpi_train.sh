#!/usr/bin/env sh
TOOLS=./build/tools

#mpiexec.hydra -prepend-rank -f machinefile -n 5 $TOOLS/caffe train \
#        --solver=examples/cifar10/cifar10_quick_solver.prototxt

mpiexec.hydra -prepend-rank -n 1 $TOOLS/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt --gpu=0 \
        : -n 1 $TOOLS/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt --gpu=1
