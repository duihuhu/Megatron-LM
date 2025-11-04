Welcome
=====

This is the code for measuring throughput under dfferent workloads for G-CRS

=====

Install
=====

The code has been tested on CUDA 6.0 to CUDA 8.0 and Ubuntu 16.04. Please set the CUDA_INSTALL_PATH variable with the directory where your CUDA resides in the makefile. 

You can do the following to see our experimental results: 
1. Run make on the directory where the code resides
2. Run script runExp.sh (You may change the configuration, this script is the case for executing on Pascal Titan X) 

Note that for a machine with multiple GPUs, set the DEVICE_SELCTED_ID in GCRSCommon.h to make the code run on the target device before run make.

Please contact through email cscjliu@comp.hkbu.edu.hk if you have any further question. 