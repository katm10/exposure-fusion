#include "quality_measures.h"
#include <timing.h>
#include <Halide.h>
#include <image_io.h>
#include <iostream>
#include <vector>

using namespace Halide;

int main(int argc, char** argv)
{
    Buffer<float> parrot = load<float>("images/parrot.png");

    // Test the intermediate debug outputs.
    Buffer<float> grayscale = Measures::compute(parrot, 1.f, 1.f, 1.f, Measures::DebugIntermediate::Grayscale);
    save(grayscale, "Output/grayscale.png");

    Buffer<float> laplacian = Measures::compute(parrot, 1.f, 1.f, 1.f, Measures::DebugIntermediate::Laplacian);
    save(laplacian, "Output/laplacian.png");

    Buffer<float> contrast = Measures::compute(parrot, 1.f, 1.f, 1.f, Measures::DebugIntermediate::Contrast);
    save(contrast, "Output/contrast.png");

    Buffer<float> saturation = Measures::compute(parrot, 1.f, 1.f, 1.f, Measures::DebugIntermediate::Saturation);
    save(saturation, "Output/saturation.png");

    Buffer<float> exposure = Measures::compute(parrot, 1.f, 1.f, 1.f, Measures::DebugIntermediate::Exposure);
    save(exposure, "Output/exposure.png");

    Buffer<float> weight = Measures::compute(parrot, 1.f, 1.f, 1.f, Measures::DebugIntermediate::Weight);

    save(weight, "Output/weight.png");

    // Test the fusion.
    
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
    save(fusion, "Output/fusion.png");

    return EXIT_SUCCESS;
}
