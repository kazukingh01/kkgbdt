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

## Setup Custom LightGBM

```bash
git clone https://github.com/kazukingh01/LightGBM.git
cd LightGBM
git submodule update --init --recursive
mkdir -p build
cd build
# sudo apt update && sudo apt install cmake
cmake ..
make -j$(nproc)
cd ../
sh ./build-python.sh install --precompile
```

別の方法

```bash
sh ./build-python.sh bdist_wheel
pip install dist/lightgbm-*.whl --force-reinstall
```

### Merge mycustom to official version

```bash
VERSION=v4.5.0
git switch master
git submodule update --init --recursive
git remote add upstream https://github.com/microsoft/LightGBM.git
git fetch upstream --tags
git branch -d my${VERSION}
git switch --detach ${VERSION}
git submodule update --init --recursive
git switch -c my${VERSION}
git cherry-pick $(git merge-base origin/master mycustom)..mycustom
git push origin --delete my${VERSION}
git push origin my${VERSION}
```

### Install cmake in Ubuntu 22.04

I got this error.

```bash
XXXXXXXXXXXXXX:~/10.git/LightGBM/build$ cmake ..
CMake Error at CMakeLists.txt:26 (cmake_minimum_required):
  CMake 3.28 or higher is required.  You are running version 3.22.1


-- Configuring incomplete, errors occurred!
```

```bash
# 1) 依存ツール
sudo apt-get update
sudo apt-get install -y ca-certificates gpg wget

# 2) 署名鍵（まだ keyring パッケージが入ってない場合だけ入れる）
test -f /usr/share/doc/kitware-archive-keyring/copyright || \
  wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
  | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

# 3) リポジトリ追加（自分のUbuntuコードネームに自動追従）
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update

# 4) keyring パッケージを入れて鍵の自動更新に切替（手動鍵は消してOK）
test -f /usr/share/doc/kitware-archive-keyring/copyright || \
  sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg
sudo apt-get install -y kitware-archive-keyring

# 5) CMake を更新
sudo apt-get install -y cmake
cmake --version
```

## Optuna dashboard

```bash
optuna-dashboard sqlite:///params_lgb.db
```

```bash
pip install psycopg2-binary==2.9.9
optuna-dashboard "postgresql+psycopg2://postgres:postgres@127.0.0.1:15432/optuna"
```
