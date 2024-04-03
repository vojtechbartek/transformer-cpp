#pragma once

#ifdef USE_CUDA
#include "../cuda/include/cuda_helpers.hpp"
namespace MatrixUtils = CudaHelpers;
#else
#include "matrix_utils.hpp"
#endif
