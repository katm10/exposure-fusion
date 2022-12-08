#include "quality_measures.h"
#include <timing.h>
#include <Halide.h>
#include <image_io.h>

using namespace Halide;

int main(int argc, char** argv)
{
    Buffer<float> parrot = load<float>("images/parrot.png");

    // Quality Measures
    {
        Buffer<float> greyscale = Measures::greyscale(parrot).realize({parrot.width(), parrot.height()});
        Buffer<float> laplacianfilter = Measures::laplacianfilter(parrot).realize({parrot.width(), parrot.height()});
        Buffer<float> absolute = Measures::absolute(laplacianfilter).realize({parrot.width(), parrot.height()});

        Buffer<float> contrast = Measures::contrast(parrot).realize({parrot.width(), parrot.height()});
        Buffer<float> saturation = Measures::saturation(parrot).realize({parrot.width(), parrot.height()});
        Buffer<float> wellexposedness = Measures::wellexposedness(parrot).realize({parrot.width(), parrot.height()});

        save(greyscale, "Output/greyscale.png");
        save(laplacianfilter, "Output/laplacianfilter.png");
        save(absolute, "Output/absolute.png");

        save(contrast, "Output/contrast.png");
        save(saturation, "Output/saturation.png");
        save(wellexposedness, "Output/wellexposedness.png");
    }

    return EXIT_SUCCESS;
}
