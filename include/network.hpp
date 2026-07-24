#ifndef CAST_NETWORK_
#define CAST_NETWORK_


#include "cast_exceptions.hpp"
#include "loss_calculator.hpp"
#include "tensor_graph/tensor_graph.hpp"

#include <memory>
#include <xtensor/containers/xarray.hpp>



namespace cast {




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
     * Holds the tensor first given to this network
     */
    std::shared_ptr<Tensor> initial_input_;

 
    /**
     * Loss metric used by this network
     */
    std::shared_ptr<LossCalculator> loss_calc_;


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
     * Sets this network's loss calculator to `calc`.
     * @param calc new loss calculator to use
     */
    void set_loss_calculator(std::shared_ptr<LossCalculator> calc) {
        //Reset the loss calculator if it exists
        if(loss_calc_) {
            loss_calc_.reset();
        }

        //Create deep pointer of the new calculator (OK for now- LossCalculators have no internal state)
        loss_calc_ = calc;
    }



    /**
     * Returns the result of the network's forward pass on `input`
     * 
     * @param input tensor to compute forward pass on
     * @return result of forward pass
     */
    xt::xarray<double> forward(xt::xarray<double> input) override {
        //Anchor the forward pass tensors by saving the input
        initial_input_ = std::make_shared<Tensor>(input);

        //Cycle the input through the operators
        std::shared_ptr<Tensor> current_node = initial_input_;
        for(const std::shared_ptr<TensorOperator>& op : operators_) {
            //Each operator creates and saves a shared pointer to its intermediate output
            current_node = op->compute_and_link(current_node);
        }

        return current_node->data();
    }

    /**
     * Computes the backward pass, initially using `predicted` and `expected`.
     * @param predicted network's prediction for a given input
     * @param expected what the network should have predicted for the input
     */
    void backward(xt::xarray<double> predicted, xt::xarray<double> expected) {
        xt::xarray<double> loss = loss_calc_->compute_gradient(predicted, expected);

        //Pass the loss backwards through each operator
        throw not_implemented();
    }

};




}
#endif