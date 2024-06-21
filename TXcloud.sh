#!/usr/bin/env bash

# !!! using source to execute this script !!!

dir=`pwd`
cd ~
#install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
bash ~/Miniconda3-py38_4.10.3-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source .bashrc

#create runtime env
yes | conda create -n pt python==3.8.10
conda activate pt
yes | conda install pytorch-gpu pytorch-lightning -c conda-forge
yes | conda install transformers
yes | pip install torchmetrics

#fix cuda library
sudo ln -isv /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.11
sudo ln -isv /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.11.0 /usr/local/cuda/extras/CUPTI/lib64/libcupti.so.11.2

#fix profile
echo -e "\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:/usr/local/cuda/extras/CUPTI/lib64" | sudo tee -a .profile
#auto activate env when login
echo -e "\nconda activate pt" | sudo tee -a .profile
source .profile

cd $dir