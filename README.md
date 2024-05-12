#transformer - cpp


### Requirements
yaml-cpp
nvcc (CUDA)

#### Installing yaml-cpp
```bash
sudo apt update
sudo apt install libyaml-cpp-dev
```

If you don't have sudo priviliges:
```bash
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
mkdir build
cd build
cmake ..
make
```

and then add to the PATH:
```bash
export CPLUS_INCLUDE_PATH=/path/to/yaml-cpp/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=/path/to/yaml-cpp/build:$LIBRARY_PATH
```

To build the CPU version, run
```bash
make
```

To build the *GPU* version, run:
```bash
make USE_CUDA=1
```


To run tests for CUDA kernels, run: 
```bash
make test USE_CUDA=1

```

To run tests for CPU version, run:
```bash
make test
```

