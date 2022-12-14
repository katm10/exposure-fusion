#include "quality_measures.h"
#include <timing.h>
#include <Halide.h>
#include <image_io.h>
#include <iostream>
#include <vector>

using namespace Halide;

int main(int argc, char** argv)
{
    // Test the different quality measures.
    {
      Buffer<float> parrot = load<float>("images/parrot.png");
      
      Buffer<float> contrast = Measures::compute(parrot, 1.f, 0.f, 0.f);
      save(contrast, "Output/parrot-contrast.png");

      Buffer<float> saturation = Measures::compute(parrot, 0.f, 1.f, 0.f);
      save(saturation, "Output/parrot-saturation.png");

      Buffer<float> exposedness = Measures::compute(parrot, 0.f, 0.f, 1.f);
      save(exposedness, "Output/parrot-exposedness.png");
    }

    // Test the fusion.
    {
      std::vector<Buffer<float>> in;
      in.push_back(load<float>("images/house-1.png"));
      in.push_back(load<float>("images/house-2.png"));
      in.push_back(load<float>("images/house-3.png"));
      in.push_back(load<float>("images/house-4.png"));

      std::vector<Buffer<float>> weight_maps;
      for (size_t i = 0; i < in.size(); i++) {
          weight_maps.push_back(Measures::compute(in[i]));
      }

      Buffer<float> fusion = Measures::compute_fusion(in, weight_maps);
      save(fusion, "Output/house-fusion.png");
    }

    {
      std::vector<Buffer<float>> in;
      in.push_back(load<float>("images/design-1.png"));
      in.push_back(load<float>("images/design-2.png"));
      in.push_back(load<float>("images/design-3.png"));
      in.push_back(load<float>("images/design-4.png"));
      in.push_back(load<float>("images/design-5.png"));
      in.push_back(load<float>("images/design-6.png"));

      std::vector<Buffer<float>> weight_maps;
      for (size_t i = 0; i < in.size(); i++) {
          weight_maps.push_back(Measures::compute(in[i]));
      }

      Buffer<float> fusion = Measures::compute_fusion(in, weight_maps);
      save(fusion, "Output/design-fusion.png");
    }
    return EXIT_SUCCESS;
}
