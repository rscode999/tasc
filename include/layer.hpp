#ifndef CNET_LAYER_
#define CNET_LAYER_

#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <cstdint>

namespace CNet {




class Layer {
public:
    xt::xarray<double> weights;

    xt::xarray<double> biases;

    int32_t n_inputs;

    int32_t n_outputs;

public:
    Layer(int32_t input_dimension, int32_t output_dimension) : n_inputs(input_dimension), n_outputs(output_dimension) {
        weights = xt::random::randn<double>({output_dimension, input_dimension}, 0.0, 1.0);
        biases = xt::random::randn<double>({output_dimension}, 0.0, 1.0);
    }

    xt::xarray<double> forward(xt::xarray<double> input) {
        return xt::linalg::dot(weights, input) + biases;
    }
};




}
#endif