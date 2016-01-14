#include "caffe/common.hpp"
#include "caffe/util/mpi.hpp"

#include <execinfo.h>
namespace caffe {

template<>
int caffe_mpi_send<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm) {
    return MPI_Send(buf, count, MPI_FLOAT, dest, tag,
                    comm);
}

template<>
int caffe_mpi_send<double>(void *buf, int count,  int dest, int tag,
                    MPI_Comm comm) {
        return MPI_Send(buf, count, MPI_DOUBLE, dest, tag,
                    comm);
}

int caffe_mpi_send(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm) {
        return MPI_Send(buf, count, datatype, dest, tag,
                    comm);
}

template<>
int caffe_mpi_recv<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
    return MPI_Recv(buf, count, MPI_FLOAT, dest, tag,
                    comm, status);
}

template<>
int caffe_mpi_recv<double>(void *buf, int count,  int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
        return MPI_Recv(buf, count, MPI_DOUBLE, dest, tag,
                    comm, status);
}

int caffe_mpi_recv(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
        return MPI_Recv(buf, count, datatype, dest, tag,
                    comm, status);
}


template <>
int caffe_mpi_isend<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
        return MPI_Isend(buf, count, MPI_FLOAT, dest, tag,comm, req);
}

template <>
int caffe_mpi_isend<double>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
        return MPI_Isend(buf, count, MPI_DOUBLE, dest, tag,comm, req);
}

int caffe_mpi_isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
        return MPI_Isend(buf, count, datatype, dest, tag,comm, req);
}

} //namespace caffe
