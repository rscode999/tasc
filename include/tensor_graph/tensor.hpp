#ifndef CAST_TENSOR_
#define CAST_TENSOR_

#include <xtensor/containers/xarray.hpp>

namespace cast {




class TensorOperator;

/**
 * Wrapper around a `xt::array` that tracks previous tensors
 */
class Tensor {
private:

    /**
     * Pointer to the operator that created this node
     */
    std::shared_ptr<TensorOperator> prev_operator_;

    /**
     * Tensor data stored inside this node
     */
    xt::xarray<double> data_;

    /**
     * Gradients of the data stored in the node
     */
    xt::xarray<double> gradients_;

    /**
    * Creates a new Tensor. All fields are uninitialized.
    * 
    * Private constructor, for use in a `TensorOperator` base class.
    */
    Tensor() = default;


public:
    friend class TensorOperator;


    /**
     * Creates a new tensor containing `input`.
     * 
     * The tensor has no registered previous operators.
     * Gradients are all 1's.
     */
    Tensor(xt::xarray<double> input) : data_(input) {
        gradients_ = xt::ones_like(input);
    }

    /**
     * @return tensor data inside this object
     */
     xt::xarray<double> data() const noexcept;

    /**
     * @return gradient of the data inside this node
     */
     xt::xarray<double> gradients() const noexcept;

};




}
#endif