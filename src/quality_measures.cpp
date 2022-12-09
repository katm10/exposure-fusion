#include "quality_measures.h"
#include "utils.h"
#include <vector>

using namespace Halide;

namespace Measures {

Buffer<float> compute(
  const Buffer<float> &in, 
  float c_weight, 
  float s_weight, 
  float e_weight,
  DebugIntermediate debug_intermediate
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

    // TODO: does the paper say a Laplacian pyramid? This is just the discrete Laplacian operator.
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

    switch (debug_intermediate) {
        case DebugIntermediate::Grayscale:
            weight = grayscale;
            break;
        case DebugIntermediate::Laplacian:
            weight = laplacian;
            break;  
        case DebugIntermediate::Contrast:
            weight = contrast;
            break;  
        case DebugIntermediate::Saturation:
            weight = saturation;
            break;  
        case DebugIntermediate::Exposure: 
            weight = exposure;
            break;
        default:
            weight(x, y) = pow(contrast(x, y), c_weight) * pow(saturation(x, y), s_weight) * pow(exposure(x, y), e_weight);
            break;
    }

    // TODO: actually schedule.
    apply_auto_schedule(weight);

    return weight.realize({in.width(), in.height()});
}

Buffer<float> compute_fusion(
  const std::vector<Buffer<float>> &in, 
  const std::vector<Buffer<float>> &weight_maps 
) {
    assert(in.size() == weight_maps.size());

    Var x("x"), y("y"), c("c");

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

    Func fusion("fusion");
    {
        fusion(x, y, c) = 0.f;
        for (size_t i = 0; i < in.size(); i++) {
            fusion(x, y, c) += in[i](x, y, c) * weight_maps[i](x, y) * normalize_weights(x, y);
        }
    }

    apply_auto_schedule(fusion);

    return fusion.realize({in[0].width(), in[0].height(), in[0].channels()});
}

} // namespace Measures