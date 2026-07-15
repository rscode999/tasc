#ifndef TAS_TENSOR_NODE_
#define TAS_TENSOR_NODE_


#include <cstdint>
#include <memory>
#include <xtensor/containers/xarray.hpp>

#include "tensor_operator.hpp"

namespace tas {




class TensorOperator;

/**
 * Wrapper around a `xt::array` that tracks previous tensors
 */
class TensorNode {
public: //normally private
    std::shared_ptr<TensorOperator> prev_operator_;

    xt::xarray<double> contents_;

    xt::xarray<double> gradients_;

    int32_t id_;

public:
    friend class TensorOperator;


    TensorNode() {
    }

    TensorNode(xt::xarray<double> input) {
        contents_ = input;
    }

    
     xt::xarray<double> contents() const noexcept {
        return contents_;
    }

     xt::xarray<double> gradients() const noexcept {
        return gradients_;
    }

    constexpr int32_t id() const noexcept {
        return id_;
    }
};




}
#endif