#include "quality_measures.h"
#include "utils.h"
#include <vector>
#include <iostream>

using namespace Halide;

namespace Measures {

constexpr int MAX_LEVELS = 20;

std::vector<float> gauss1DFilterValues(float sigma, float truncate)
{
  // Create the 1D Gaussian kernel.
  float total = 0;
  std::vector<float> gauss1DFilter;
  for (int i = ceil(-truncate * sigma); i <= ceil(truncate * sigma); i++){
    gauss1DFilter.push_back(exp(-0.5 * pow(i / sigma, 2)));
    total += gauss1DFilter.back();
  }

  for (size_t i = 0; i < gauss1DFilter.size(); i++){
    gauss1DFilter[i] /= total;
  }

  return gauss1DFilter;
}

Func upsample(Func input)
{
  // Use bilinear interpolation to upsample an image. 
  Func upx("upx"), upy("upy");
  Var x("x"), y("y");

  upx(x, y) = lerp(input((x + 1) / 2, y), input((x - 1) / 2, y), ((x % 2) * 2 + 1) / 4.f);
  upy(x, y) = lerp(upx(x, (y + 1) / 2), upx(x, (y - 1) / 2), ((y % 2) * 2 + 1) / 4.f);

  return upy;
}

Func downsample(Func input) 
{
  // Downsample with [1, 3, 3, 1] filter
  Func downx("downx"), downy("downy");
  Var x("x"), y("y");

  downx(x, y) = (input(x * 2 - 1, y) + input(x * 2, y) * 3 + input(x * 2 + 1, y) * 3 + input(x * 2 + 2, y)) / 8.f;
  downy(x, y) = (downx(x, y * 2 - 1) + downx(x, y * 2) * 3 + downx(x, y * 2 + 1) * 3 + downx(x, y * 2 + 2)) / 8.f;

  return downy;
}

Buffer<float> compute(
  const Buffer<float> &in, 
  float c_weight, 
  float s_weight, 
  float e_weight
) {
    Var x("x"), y("y"), c("c");

    // Clamp the input so bounds inference can infer all the rest (i.e. for laplacian).
    // This maybe isn't the most optimal thing, but it's the easiest.
    Func input("input");
    input(x, y, c) = in(clamp(x, 0, in.width() - 1), clamp(y, 0, in.height() -1), c);

    // Compute luminance for laplacian.
    Func grayscale("grayscale");
    float lum_weights[3] = {0.299, 0.587, 0.114};
    grayscale(x, y) = input(x, y, 0) * lum_weights[0] + input(x, y, 1) * lum_weights[1] + input(x, y, 2) * lum_weights[2];

    Func laplacian("laplacian");
    float lap_weights[3][3] = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
    laplacian(x, y) = cast<float>(lap_weights[0][0] * grayscale(x - 1, y - 1) + lap_weights[0][1] * grayscale(x, y - 1) + lap_weights[0][2] * grayscale(x + 1, y - 1) 
                                  + lap_weights[1][0] * grayscale(x - 1, y) + lap_weights[1][1] * grayscale(x, y) + lap_weights[1][2] * grayscale(x + 1, y) 
                                  + lap_weights[2][0] * grayscale(x - 1, y + 1) + lap_weights[2][1] * grayscale(x, y + 1) + lap_weights[2][2] * grayscale(x + 1, y + 1));

    // Compute contrast weight.
    Func contrast("contrast");
    contrast(x, y) = abs(laplacian(x, y));

    // Now compute saturation weight.
    Func saturation("saturation");
    {
        Expr R = input(x, y, 0);
        Expr G = input(x, y, 1);
        Expr B = input(x, y, 2);
        Expr mu = (R + G + B) / 3.f;
        saturation(x, y) = sqrt((pow(R - mu, 2.f) + pow(G - mu, 2.f) + pow(B - mu, 2.f)) / 3.f);
    }

    // Now compute exposure weight.
    Func exposure("exposure");
    {
        const float sigma = 0.2f; // from paper.
        Expr R = exp(cast<double>(-0.5f * pow(input(x, y, 0) - 0.5f, 2)) / std::pow(sigma, 2.f));
        Expr G = exp(cast<double>(-0.5f * pow(input(x, y, 1) - 0.5f, 2)) / std::pow(sigma, 2.f));
        Expr B = exp(cast<double>(-0.5f * pow(input(x, y, 2) - 0.5f, 2)) / std::pow(sigma, 2.f));
        exposure(x, y) = cast<float>(R * G * B);
    }

    Func weight("weight");
    weight(x, y) = pow(contrast(x, y), c_weight) * pow(saturation(x, y), s_weight) * pow(exposure(x, y), e_weight);

    // TODO: actually schedule.
    apply_auto_schedule(weight);

    return weight.realize({in.width(), in.height()});
}

Buffer<float> compute_fusion(
  const std::vector<Buffer<float>> &in, 
  const std::vector<Buffer<float>> &weight_maps 
) {
    assert(in.size() == weight_maps.size());

    Var x("x"), y("y"), c("c"), k("k");

    Func normalize_weights("normalize_weights");
    {
        Func sum_weights("sum_weights");
        {
          sum_weights(x, y) = 0.f;
          for (size_t i = 0; i < weight_maps.size(); i++) {
              sum_weights(x, y) += weight_maps[i](x, y);
          }
        }

        normalize_weights(x, y) = 1.f / sum_weights(x, y);
    }

    // Func fusion("fusion");
    // {
    //     fusion(x, y, c) = 0.f;
    //     for (size_t i = 0; i < in.size(); i++) {
    //         fusion(x, y, c) += in[i](x, y, c) * weight_maps[i](x, y) * normalize_weights(x, y);
    //     }
    // }

    // apply_auto_schedule(fusion);
    
    // return fusion.realize({in[0].width(), in[0].height(), in[0].channels()});

    int levels = (int) log2(std::min(in[0].width(), in[0].height())) - 1;
    std::cout << "levels: " << levels << std::endl;

    Func weightGaussian0[in.size()]; // Holds the previous gaussian result
    Func weightGaussian1[in.size()]; // Holds the current gaussian result

    Func inputGaussian0[in.size()]; // Holds the previous gaussian result
    Func inputGaussian1[in.size()]; // Holds the current gaussian result
    Func inputLaplacian[in.size()]; // Holds the current laplacian result

    for (size_t i = 0; i < in.size(); i++) {
      weightGaussian0[i](x, y) = weight_maps[i](x, y) * normalize_weights(x, y);
      weightGaussian1[i](x, y) = downsample(weightGaussian0[i])(x, y);

      inputGaussian0[i](x, y) = in[i](x, y, 0);
      inputGaussian1[i](x, y) = downsample(inputGaussian0[i])(x, y);
      inputLaplacian[i](x, y) = weightGaussian0[i](x, y) - upsample(weightGaussian1[i])(x, y);
    }

    Func combined[levels];
    for (int j = 0; j < levels; j++) {
      combined[j](x, y) = 0.f;
      for (size_t i = 0; i < in.size(); i++) {
        combined[j](x, y) += inputLaplacian[i](x, y) * weightGaussian0[i](x, y);

        weightGaussian0[i](x, y) = weightGaussian1[i](x, y);
        weightGaussian1[i](x, y) = downsample(weightGaussian0[i])(x, y);

        inputGaussian0[i](x, y) = inputGaussian1[i](x, y);
        inputGaussian1[i](x, y) = downsample(inputGaussian0[i])(x, y);
        inputLaplacian[i](x, y) = weightGaussian0[i](x, y) - upsample(weightGaussian1[i])(x, y);
      }
    }

    Func final[levels];
    final[levels - 1](x, y) = combined[levels -1](x, y);
    for (int j = levels - 2; j >= 0; --j) {
      final[j](x, y) = combined[j](x, y) + upsample(final[j + 1])(x, y);
    }

    apply_auto_schedule(final[0]);
    return final[0].realize({in[0].width(), in[0].height(), in[0].channels()});

    // Func finalLaplPyr[levels][in[0].channels()];
    // for (int j = 0; j < levels; ++j) {
    //   for (int c = 0; c < in[0].channels(); ++c) {
    //     for (size_t i = 0; i < in.size(); ++i) {
    //       Func input("input");
    //       Func weight("weight");
    //       input(x, y) = in[i](x, y, c);
    //       weight(x, y) = weight_maps[i](x, y) * normalize_weights(x, y);

    //       Func gaussPyr[levels];
    //       gaussPyr[0](x, y) = weight(x, y);
    //       for (int i = 1; i < levels; i++) {
    //           gaussPyr[i](x, y) = downsample(gaussPyr[i - 1])(x, y);
    //       }

    //       Func inGaussPyr[levels];
    //       inGaussPyr[0](x, y) = input(x, y);
    //       for (int i = 1; i < levels; i++) {
    //           inGaussPyr[i](x, y) = downsample(inGaussPyr[i - 1])(x, y);
    //       }

    //       Func inLaplPyr[levels];
    //       inLaplPyr[levels - 1](x, y) = inGaussPyr[levels - 1](x, y);
    //       for (int i = levels - 2; i >= 0; i--) {
    //         inLaplPyr[i](x, y) = inGaussPyr[i](x, y) - upsample(inGaussPyr[i + 1])(x, y);
    //       }

    //       finalLaplPyr[j][c](x, y) += gaussPyr[j](x, y) * inLaplPyr[j](x, y);
    //     }
    //   }
    // }

    // Func combinePyr[levels][in[0].channels()];
    // for (int c = 0; c < in[0].channels(); ++c) {
    //   combinePyr[levels - 1][c](x, y) = finalLaplPyr[levels - 1][c](x, y);
    // }
    // for (int j = levels - 2; j >= 0; --j) {
    //   for (int c = 0; c < in[0].channels(); ++c) {
    //     combinePyr[j][c](x, y) = finalLaplPyr[j][c](x, y) + upsample(combinePyr[j + 1][c])(x, y);
    //   }
    // }

    // Func sum("sum");
    // {
    //   sum(x, y, c) = 0.f;
    //   for (int c = 0; c < in[0].channels(); ++c) {
    //     sum(x, y, c) += combinePyr[0][c](x, y);
    //   }
    // }

    // apply_auto_schedule(sum);

    // return sum.realize({in[0].width(), in[0].height(), in[0].channels()});
}

} // namespace Measures