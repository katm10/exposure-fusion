#pragma once

#include <iostream>
#include <cmath>
#include <Halide.h>

using Halide::Func;
using Halide::Buffer;

namespace Measures {

  Func greyscale(const Buffer<float>& input);
  Func laplacianfilter(const Buffer<float>& input);
  Func absolute(const Buffer<float>& input);

  Func contrast(const Buffer<float>& input);
  Func saturation(const Buffer<float>& input);
  Func wellexposedness(const Buffer<float>& input, float sigma = 0.2f);

} // namespace Measures
