#include "caffe/util/mpijob.hpp"
#include "caffe/common.hpp"

namespace caffe {

shared_ptr<MPIComm> MPIComm::singleton_;

MPIComm::MPIComm():
  running_(false), started_(false),ready_(false){}

MPIComm::~MPIComm() {
  if(IsRunning()){
    EndProcessing();
  }
}

bool MPIComm::IsRunning(){
  return running_.load();
}

bool MPIComm::IsIdle(){
  mutex::scoped_lock lock(queue_mutex_);
  return task_queue_.empty();
}

void MPIComm::WaitAll() {
  mutex::scoped_lock lock(queue_mutex_);
  while (task_queue_.size()){
    DLOG(INFO)<<"Waiting for tasks to finish, task size "<<task_queue_.size()<<"\n";
    cond_finish_.wait(lock);
  }
  DLOG(INFO)<<"all task done on "<<Caffe::MPI_my_rank()<<"\n";
}

void MPIComm::StartProcessing() {

  running_.store(true);
  // start the transmission thread
  try {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    thread_.reset(
        new boost::thread(&MPIComm::ThreadFunc, this,device));
  } catch (...) {
    LOG(FATAL)<<"Cannot start MPI comminication thread";
  }
}

void MPIComm::EndProcessing(){
  if (IsRunning()) {
    try {
      cond_work_.notify_one();
      running_.store(false); //notify the transmission thread to finish and shutdown
      thread_->join();
    } catch (...) {
      LOG(FATAL)<<"Cannot destroy MPI comminication thread";
    }
  }
}

void MPIComm::AddJob(MPIJob new_job) {
  if (IsRunning()) {
    while(!started_.load());
    mutex::scoped_lock lock(queue_mutex_);
    DLOG(INFO) << "adding job on " << Caffe::MPI_my_rank() << " task queue size " << task_queue_.size() << " \n";
    task_queue_.push(new_job);
    lock.unlock();
    cond_work_.notify_one();
  }else{
    LOG(FATAL)<<"Cannot push job while MPI Comm is shutting down";
  }
}

void MPIComm::DispatchJob(MPIJob &job) {
  MPI_Datatype data_type = (job.dtype_size_ == 4) ? MPI_FLOAT : MPI_DOUBLE;

  // call MPI APIs for real works
  switch (job.op_) {
    case OP_SUM_ALL: {
      DLOG(INFO)<<"Running all reduce\n";
      MPI_Allreduce((job.src_ptr_ == job.dst_ptr_) ? MPI_IN_PLACE : job.src_ptr_,
                              job.dst_ptr_, job.count_, data_type,
                              MPI_SUM, MPI_COMM_WORLD
      );
      break;
    }
    case OP_BROADCAST: {
      CHECK_EQ(job.src_ptr_, job.dst_ptr_);
      MPI_Bcast(job.src_ptr_, job.count_, data_type,
                          0, MPI_COMM_WORLD);
      break;
    }
    default: {
      LOG(FATAL)<<"Unknown MPI job type";
    }
  }
}

void MPIComm::ThreadFunc(int device){
#ifndef CPU_ONLY
  LOG(ERROR)<<"device_id is "<<device;
  CUDA_CHECK(cudaSetDevice(device));
#endif
  started_.store(true);
  MPIJob job;
  while (true){
    mutex::scoped_lock lock(queue_mutex_);
    while( task_queue_.empty() && IsRunning()){
      DLOG(INFO)<<"no job running, waiting on cond";
      cond_work_.wait(lock);
    }
    lock.unlock();

    DLOG(INFO)<<"Cond fulfilled, dispatching job";
    if (IsRunning()){
      job = task_queue_.front();
      DLOG(INFO)<<task_queue_.size();
      DispatchJob(job);
      mutex::scoped_lock pop_lock(queue_mutex_);
      task_queue_.pop();
      pop_lock.unlock();
      cond_finish_.notify_one();
      DLOG(INFO)<<"job finished, poped taskqueue";
    }else{
      break;
    }

  }

  // finish remaining jobs
  while (!task_queue_.empty()){
    boost::lock_guard<mutex> lock(queue_mutex_);
    job = task_queue_.front();
    task_queue_.pop();
    DispatchJob(job);
  }
}




};
