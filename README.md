这是我从2015年12月BVLC/caffe fork 过来的。根据google的Parameter server的原理并参考sailorsb/caffe-parallel 使用mpi实现的一个多机多卡版本。

具体改动：
solver.cpp 中加入mpi相关的东西
每个进程相同的模型，加载不同的数据进行训练。每次计算的梯度（diff）都会发送给rank 0 的进程，rank 0 对所有的diff求平均最为最终的diff。
rank 0 更新自己的模型，并将更新后的模型分发给其他模型副本就是其他进程。再开始新一轮的计算，这样实现数据的并行化。其中rank 0进程充当parameter server。
模型更新是采取同步的方式进行的，需要所有的都计算完并发送计算的结果后才能继续。

dater_reader.cpp 中实现每个模型副本数据加载的不同，每个进程根据自己的rank 号获取leveldb或者lmdb中的部分数据。故进行训练之前，每个机器上都需要存在一份数据。

关于 gtest部分的代码，我没有写...如果runtest 可能会出错 ps：我没去试过。

To Do List：
1.小网络时网络传输消耗不是很大，大网络时，每个进程都会发送数据给rank 0，网络延迟较大。这个问题将进行优化。
2.目前还只是支持CNN，关于RNN还未支持，很快将加入LSTM的实现。
3.目前使用gpu时还需要自己手动的指定gpu 编号

本人的实际的机器使用：
2机器共4个GPU
操作系统 centos 6.7
MPI框架：mpich3
Atlas

安装：
1.按照caffe的官网说明正常安装caffe所有的环境，并安装mpi相关的软件。本人使用MPICH3
2.编译caffe，参考caffe的说明，修改Makefile.config文件，CUSTOM_CXX := /your/mpi/bin/mpic++该选项改为自己的mpi路径

程序运行：
以cifar10为例子，先进行数据的准本在caffe-parallel的根目录下 sh data/cifar10/get_cifar10.sh 下载数据。
再sh examples/cifar10/create_cifar10.sh 将数据转换为lmdb经计算图片的均值
运行 sh examples/cifar10/mpi_train.sh 开始进行训练
脚本中是一个机器上的两个gpu并行，多机器之间可以修改为如下部分
 -n 1 -host node1 $TOOLS/caffe train --solver=models/bvlc_alexnet/solver.prototxt --gpu=0 
在node1机器上启动一个caffe的进程，使用gpu 0。node1的ip应该存在本机的hosts文件中。推荐看下mpich3的官方文档，将该部分写到一个machinfile中，运行时调用machinefile

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
