[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <h3 align="center">TVM TensorRT Deployment Performance Comparison</h3>
</div>


[contributors-shield]: https://img.shields.io/github/contributors/jc-su/tvm_tensorrt_comparsion.svg?style=for-the-badge
[contributors-url]: https://github.com/jc-su/tvm_tensorrt_comparsion/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/jc-su/tvm_tensorrt_comparsion.svg?style=for-the-badge
[forks-url]: https://github.com/jc-su/tvm_tensorrt_comparsion/network/members
[stars-shield]: https://img.shields.io/github/stars/jc-su/tvm_tensorrt_comparsion.svg?style=for-the-badge
[stars-url]: https://github.com/jc-su/tvm_tensorrt_comparsion/stargazers
[issues-shield]: https://img.shields.io/github/issues/jc-su/tvm_tensorrt_comparsion.svg?style=for-the-badge
[issues-url]: https://github.com/jc-su/tvm_tensorrt_comparsion/issues
[license-shield]: https://img.shields.io/github/license/jc-su/tvm_tensorrt_comparsion.svg?style=for-the-badge
[license-url]: https://github.com/jc-su/tvm_tensorrt_comparsion/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/jcsu/

# Install the dependencies

```bash
sudo apt install opencv
```
Installing TensorRT from [this link](https://developer.nvidia.com/tensorrt)
Installing TVM form [TVM tutorial](https://tvm.apache.org/docs/install/from_source.html)
Make sure to choose set(USE_LLVM ON) and set(USE_CUDA ON) in build/config.cmake

# How to build
```bash
git clone --recursive https://github.com/jc-su/tvm_tensorrt_comparsion
mkdir build
cd build
cmake ..
make $(nproc)
```
