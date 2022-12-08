#include "quality_measures.h"
#include "utils.h"

using namespace Halide;

namespace Measures {

Func greyscale(const Buffer<float>& input) {
  Func color2grey("color2grey");
  Var x("x"), y("y"), c("c");

  std::vector<float> weights = {0.299, 0.587, 0.114};
  color2grey(x, y) = input(x, y, 0) * weights[0] + input(x, y, 1) * weights[1] + input(x, y, 2) * weights[2];
  apply_auto_schedule(color2grey);
  return color2grey;
}

Func laplacianfilter(const Buffer<float>& input) {
  float kernel[3][3] = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};

  Func clamped("clamped");
  Var x("x"), y("y");
  clamped(x, y) = cast<float>(input(clamp(x,0,input.width() - 1), clamp(y, 0, input.height() -1)));

  Func laplacian("laplacian");

  laplacian(x, y) = cast<float>(kernel[0][0] * clamped(x - 1, y - 1) + kernel[0][1] * clamped(x, y - 1) + kernel[0][2] * clamped(x + 1, y - 1) 
                              + kernel[1][0] * clamped(x - 1, y) + kernel[1][1] * clamped(x, y) + kernel[1][2] * clamped(x + 1, y) 
                              + kernel[2][0] * clamped(x - 1, y + 1) + kernel[2][1] * clamped(x, y + 1) + kernel[2][2] * clamped(x + 1, y + 1));

  
  apply_auto_schedule(laplacian);
  return laplacian;
}

Func absolute(const Buffer<float>& input) {
  Func abs("abs");
  Var x("x"), y("y");

  abs(x, y) = abs(input(x, y));
  apply_auto_schedule(abs);
  return abs;
}

Func contrast(const Buffer<float>& input){
  Func contrast("contrast");
  Var x("x"), y("y");

  contrast(x,y) = absolute(
    laplacianfilter(
      greyscale(input).realize({input.width(), input.height()})
    ).realize({input.width(), input.height()})
  );

  apply_auto_schedule(contrast);
  return contrast;
}

Func saturation(const Buffer<float>& input) {
  Func sat("saturation");
  Var x("x"), y("y");

  Expr R = input(x, y, 0);
  Expr G = input(x, y, 1);
  Expr B = input(x, y, 2);

  Expr mu = (R + G + B) / 3;

  sat(x, y) = sqrt((pow(R - mu, 2) + pow(G - mu, 2) + pow(B - mu, 2)) / 3);
  apply_auto_schedule(sat);
  return sat;
}

Func wellexposedness(const Buffer<float>& input, float sigma){
  Func exposed("exposed");
  Var x("x"), y("y");

  Expr R = exp(cast<double>(-0.5f * pow(input(x, y, 0) - 0.5f, 2)) / pow(sigma, 2));
  Expr G = exp(cast<double>(-0.5f * pow(input(x, y, 1) - 0.5f, 2)) / pow(sigma, 2));
  Expr B = exp(cast<double>(-0.5f * pow(input(x, y, 2) - 0.5f, 2)) / pow(sigma, 2));

  exposed(x, y) = cast<float>(R * G * B);
  apply_auto_schedule(exposed);
  return exposed;
}

} // namespace Measures
