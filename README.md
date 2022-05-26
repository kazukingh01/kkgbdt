# kkgbdt

## Setup LightGBM GPU
see: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
```bash
sudo docker pull nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
sudo docker run -itd --gpus all --name cuda --net=nw -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --device /dev/dri -v /home/share:/home/share -v /home/backup:/home/backup --shm-size=10g nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04 /bin/bash --login
sudo docker exec -it cuda /bin/bash
apt-get update
apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv/plugins/python-build
./install.sh
/usr/local/bin/python-build -v 3.8.12 ~/local/python-3.8.12
echo 'export PATH="$HOME/local/python-3.8.12/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
pip install virtualenv
cd ~
virtualenv venv
source venv/bin/activate
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
make -j$(nproc)
cd ..
cd python-package/
python setup.py install --precompile
cd ..
mkdir -p /etc/OpenCL/vendors && echo "http://libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd # see: https://github.com/microsoft/LightGBM/issues/586
```