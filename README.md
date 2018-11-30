# GeePS

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

[GeePS](https://cuihenggang.github.io/archive/paper/[eurosys16]geeps.pdf) is a parameter server library that scales single-machine GPU machine learning applications (such as Caffe) to a cluster of machines.


## Download and build GeePS and Caffe application on Centos 7

```
sudo yum install boost-devel gflags-devel zeromq-devel glog-devel protobuf-devel atlas-devel cppzmq-devel opencv-devel hdf5-devel leveldb-devel lmdb-devel
git clone https://github.com/niketanpansare/geeps.git
scons -j8
cd geeps/apps
git clone https://github.com/cuihenggang/caffe.git
cd caffe
git checkout 4c16192739df54c971dfccaf871fdf896be237b9 .
make -j8
```

If you have latest version of CuDNN, you may want to copy [the latest CuDNN header](https://raw.githubusercontent.com/BVLC/caffe/master/include/caffe/util/cudnn.hpp) to `apps/caffe/util/cudnn.hpp`.

If you are getting an error `cannot find -lcblas or -latlas`, you may want to do:
```
ln -s /usr/lib64/atlas/libtatlas.so /usr/lib64/libatlas.so
ln -s /usr/lib64/atlas/libsatlas.so /usr/lib64/libcblas.so
```

## Download and build GeePS and Caffe application

Run the following command to download GeePS and (our slightly modified) Caffe:

```
git clone --recurse-submodules https://github.com/cuihenggang/geeps.git
```

If you use the Ubuntu 14.04 system, you can run the following commands (from geeps root directory) to install the dependencies:

```
./scripts/install-geeps-deps-ubuntu14.sh
./scripts/install-caffe-deps-ubuntu14.sh
```

Also, please make sure your CUDA library is installed in `/usr/local/cuda`.

After installing the dependencies, you can build GeePS by simply running this command from geeps root directory:

```
scons -j8
```

You can then build (our slightly modified) Caffe by first entering the `apps/caffe` directory and then running `make -j8`:

```
cd apps/caffe
make -j8
```


## Caffe's CIFAR-10 example on two machines

You can run Caffe distributedly across a cluster of machines with GeePS. In this section, we will show you the steps to run Caffe's CIFAR-10 example on two machines.

All commands in this section are executed from the `apps/caffe` directory:

```
cd apps/caffe
```

You will first need to prepare a machine file as `examples/cifar10/2parts/machinefile`, with each line being the host name of one machine. Since we use two machines in this example, this machine file should have two lines, such as:

```
host0
host1
```

We will use `pdsh` to launch commands on those machines with the `ssh` protocol, so please make sure that you can `ssh` to those machines without password.

When you have your machine file in ready, you can run the following command to download and prepare the CIFAR-10 dataset:

```
./data/cifar10/get_cifar10.sh
./examples/cifar10/2parts/create_cifar10_pdsh.sh
```

Our script will partition the datasets into two parts, one for each machine. You can then train an Inception network on it with this command:

```
./examples/cifar10/2parts/train_inception.sh
```

Please look at our [wiki](https://github.com/cuihenggang/geeps/wiki) for more details. Happy training!


## Automatic training hyperparameter tuning

[MLtuner-GeePS](https://github.com/cuihenggang/mltuner-geeps) is an extended version of GeePS with automatic training hyperparameter tuning support. It includes a lightweight [MLtuner](https://cuihenggang.github.io/archive/paper/[arxiv]mltuner.pdf) module that automatically tunes the training hyperparameters for distributed ML training (including learning rate, momentum, batch size, data staleness, etc).


## Reference Paper

Henggang Cui, Hao Zhang, Gregory R. Ganger, Phillip B. Gibbons, and Eric P. Xing.
[GeePS: Scalable Deep Learning on Distributed GPUs with a GPU-Specialized Parameter Server](https://cuihenggang.github.io/archive/paper/[eurosys16]geeps.pdf).
In ACM European Conference on Computer Systems, 2016 (EuroSys'16).

Henggang Cui, Gregory R. Ganger, and Phillip B. Gibbons.
[MLtuner: System Support for Automatic Machine Learning Tuning](https://cuihenggang.github.io/archive/paper/[arxiv]mltuner.pdf).
arXiv preprint 1803.07445.

