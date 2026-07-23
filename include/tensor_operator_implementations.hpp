#ifndef CAST_TENSOR_OPERATOR_
#define CAST_TENSOR_OPERATOR_

#include "tensor_graph_base.hpp"
#include "cast_exceptions.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <cassert>
#include <cstdint>


namespace cast {

    




class Linear1d : public TensorOperator {
private:
    /**
    * Required size of input vectors
    */
    int32_t input_dimension_;

    /**
    * Size of vectors coming from the linear forward operation
    */
    int32_t output_dimension_;
    
    /**
    * 2d weight matrix
    */
    xt::xarray<double> weights_;

    /**
    * 1d bias vector
    */
    xt::xarray<double> biases_;

    /**
    * Value from this layer's output
    */
    xt::xarray<double> stored_output_;

public:
    /**
    * Creates a 1d linear layer with `input_dimension` inputs and `output_dimension` outputs
    * @param input_dimension required size of input vectors. Precondition: Positive
    * @param output_dimension size of output vectors. Precondition: Positive
    */
    Linear1d(int32_t input_dimension, int32_t output_dimension) : input_dimension_(input_dimension), output_dimension_(output_dimension) {
        assert(input_dimension > 0);
        assert(output_dimension > 0);
        weights_ = xt::random::randn<double>({output_dimension, input_dimension}, 0, 1);
        biases_ = xt::random::randn<double>({output_dimension}, 0, 1);
    }

    /**
    * Returns the result of the forward pass on 
    */
    xt::xarray<double> compute(xt::xarray<double> input) const override {
        assert(input.dimension() == 1 && "Input must be a vector");
        assert(input.size() == input_dimension_ && "Input tensor must have dimension matching the layer's input dimension");
        return xt::linalg::dot(weights_, input) + biases_;
    }

    xt::xarray<double> compute_backwards_pass(xt::xarray<double> input) const override {
        throw not_implemented();
    }
};



}
#endif