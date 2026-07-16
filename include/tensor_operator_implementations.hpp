#ifndef TASC_TENSOR_OPERATOR_
#define TASC_TENSOR_OPERATOR_


#include <memory>
#include <string>
#include <vector>
#include "tensor_graph_base.hpp"


namespace tasc {

    




/**
 * Temporary operator that adds 1 to each element of its input
 */
class Increment : public TensorOperator {
public:
    /**
     * Adds 1 to each element of `input`.
     */
    xt::xarray<double> compute(xt::xarray<double> input) override {
        xt::xarray<double> out = input;
        for(double& i : out) {
            i += 1;
        }
        return out;
    }

    /**
     * @return the string "increment"
     */
    std::string name() override {
        return "increment";
    }
};


/**
 * Multiplies each element of its input by 2
 */
class TimesTwo : public TensorOperator {
public:
    /**
     * Multiplies each element of `input` by 2.
     */
    xt::xarray<double> compute(xt::xarray<double> input) override {
        xt::xarray<double> out = input;
        for(double& i : out) {
            i *= 2;
        }
        return out;
    }

    /**
     * @return the string "add_five"
     */
    std::string name() override {
        return "add_five";
    }
};



}
#endif