#ifndef TASC_NETWORK_
#define TASC_NETWORK_


#include <cstdint>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <xtensor/containers/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "tasc_exceptions.hpp"
#include "tensor_operator_implementations.hpp"


namespace tasc {




/**
 * Network that allows users to arrange layers and operators in any order
 */
class CustomNetwork {
public:
    /**
     * Creates an empty custom network object
     */
    CustomNetwork() {
    }


    /**
     * Returns the result of the network's forward pass.
     * 
     * This method is user-defined.
     */
    virtual xt::xarray<double> forward(xt::xarray<double> input) = 0;
};



/**
 * (Sequentially defined) network
 */
class Network : public CustomNetwork {
public: //normally private
    /**
     * Operators that the network uses, in order
     */
    std::vector<std::shared_ptr<TensorOperator>> operators_;

    /**
     * Holds the first tensor given to this network
     */
    std::shared_ptr<TensorNode> initial_input_;

public:
    /**
     * Creates a new network
     */
    Network() = default;

    /**
     * Registers `op` as the next operator to execute in the network
     * @param op new operator to add
     */
    void add_operator(std::shared_ptr<TensorOperator> op) {
        operators_.push_back(op);
    }

    /**
     * Returns the result of the network's forward pass on `input`
     * 
     * @param input tensor to compute forward pass on
     * @return result of forward pass
     */
    xt::xarray<double> forward(xt::xarray<double> input) override {
        //Anchor the forward pass tensors by saving the input
        initial_input_ = std::make_shared<TensorNode>(input);

        //Cycle the input through the operators
        std::shared_ptr<TensorNode> current_node = initial_input_;
        for(const std::shared_ptr<TensorOperator>& op : operators_) {
            //Each operator creates and saves a shared pointer to its intermediate output
            current_node = op->compute_and_link(current_node);
        }

        return current_node->data();
    }

    /**
     * Exports `net` to the output stream `output_stream`, returning `output_stream` with `net` inside
     * @param output_stream stream to export the network to
     * @param net network object to export
     * @return `output_stream` with `net`
     */
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Network& net);
    
};


template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const Network& net) {
    for(std::shared_ptr<TensorOperator> op_ptr : net.operators_) {
        output_stream << op_ptr->name() << '\n';
    }

    return output_stream;
}




}
#endif