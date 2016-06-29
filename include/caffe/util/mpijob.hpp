#ifndef CAFFE_MPIJOB_HPP
#define CAFFE_MPIJOB_HPP

#include <mpi.h>
#include <boost/shared_ptr.hpp>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <queue>

using std::queue;
using boost::mutex;
using boost::condition_variable;
using boost::shared_ptr;
using boost::atomic;

namespace caffe { 

enum OperationType {
    OP_SUM_ALL,OP_BROADCAST,OP_GATHER,OP_SCATTER
};

class MPIJob {
 public:
  void* src_ptr_;
  void* dst_ptr_;
  int count_;
  int dtype_size_;
  OperationType op_;
};

class MPIComm{
  public:
    ~MPIComm();
    inline static MPIComm& Get() {
      if (!singleton_.get()) {
        singleton_.reset(new MPIComm());
        singleton_->StartProcessing();
      }
      return *singleton_;
    }

    inline static void AddMPIJob(MPIJob job){ Get().AddJob(job);};
    inline static void Syncrhonize(){Get().WaitAll();}
    inline static bool IsReady(){return Get().ready_;}
    inline static void SetReady(bool ready){Get().ready_=ready;}

  private:
    MPIComm();

    void ThreadFunc(int device);
    void DispatchJob(MPIJob& job);
    bool IsRunning();
    bool IsIdle();
    void StartProcessing();
    void EndProcessing();
    void AddJob(MPIJob new_job);
    void WaitAll();

    queue<MPIJob> task_queue_;
    mutable mutex queue_mutex_;
    atomic<bool> running_, started_;
    shared_ptr<boost::thread> thread_;
    condition_variable cond_work_;
    condition_variable cond_finish_;
    bool ready_;

    static shared_ptr<MPIComm> singleton_;

};

};//namespace caffe

#endif //CAFFE_MPIJOB_HPP 
