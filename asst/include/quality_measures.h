#pragma once

#include <iostream>
#include <cmath>
#include <Halide.h>

using Halide::Func;
using Halide::Buffer;

namespace Measures {

  Buffer<float> compute(
    const Buffer<float> &in, 
    float c_weight = 1.f, 
    float s_weight = 1.f, 
    float e_weight = 1.f
  );

  Buffer<float> compute_fusion(
  const std::vector<Buffer<float>> &in, 
  const std::vector<Buffer<float>> &weight_maps 
);

} // namespace Measures
