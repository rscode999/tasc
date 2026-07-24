#include "tensor.hpp"


namespace cast {




xt::xarray<double> Tensor::data() const noexcept {
    return data_;
}


xt::xarray<double> Tensor::gradients() const noexcept {
    return gradients_;
}




}