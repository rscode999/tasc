#ifndef TAS_NETWORK_
#define TAS_NETWORK_


#include <cstdint>
#include <memory>
#include <unordered_map>
#include <xtensor/containers/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "tas_exceptions.hpp"
#include "tensor_node.hpp"


namespace tas {




class CustomNetwork {
private:

    /**
     * The next layer added to this network will be assigned this ID
     */
    int32_t next_id_;


public:
    /**
     * Creates an empty network object
     */
    CustomNetwork() {
        next_id_ = 0;
    }


    virtual TensorNode forward(TensorNode input) = 0;
};



/**
 * (Sequentially defined) network
 */
class Network : public CustomNetwork {
private:
    std::vector<std::shared_ptr<TensorOperator>> operators;

public:

    void add_operator(std::shared_ptr<TensorOperator> op) {
        operators.push_back(op);
    }


    xt::xarray<double> forward(xt::xarray<double> input) {
        xt::xarray<double> out = input;
        for(const std::shared_ptr<TensorOperator>& op : operators) {
            out = op->compute(out);
        }

        return out;
    }
};




}
#endif