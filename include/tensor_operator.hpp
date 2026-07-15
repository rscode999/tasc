#ifndef TAS_TENSOR_OPERATOR_
#define TAS_TENSOR_OPERATOR_


#include <memory>
#include <string>
#include <vector>

#include "tensor_node.hpp"



namespace tas {

    


class TensorNode;


class TensorOperator : public std::enable_shared_from_this<TensorOperator> {
public: //normally protected
    /**
     * Pointers to all tensors that are computed, just prior to this operator receiving them
     */
    std::vector<std::shared_ptr<TensorNode>> predecessors_ = {};

    /**
     * Pointers to all tensors that are the direct result of this operation
     */
    std::vector<std::weak_ptr<TensorNode>> successors_ = {}; //Must use weak pointers, to avoid dependency traps

    TensorOperator() = default;

public:


    /**
     * Returns a shared pointer to a tensor node, representing the output of this operator.
     * 
     * The node given to this method is registered as a predecessor to this operator.
     */
    std::shared_ptr<TensorNode> compute_and_link(std::shared_ptr<TensorNode> input_node) {
        //Register the input node as a predecessor
        predecessors_.push_back(input_node);

        //Carry out the operation
        xt::xarray<double> output_tensor = compute(input_node->contents());

        //Wrap operation in tensor node
        TensorNode output_node;
        output_node.contents_ = output_tensor;
        output_node.gradients_ = xt::zeros<double>(output_tensor.shape());

        std::shared_ptr<TensorNode> output_node_ptr = std::make_shared<TensorNode>(std::move(output_node));
        //Register this operation object as the new output's precedessor
        output_node_ptr->prev_operator_ = shared_from_this();

        return output_node_ptr;
    }

public:
    template <typename ConcreteType, typename... Args>
    static std::shared_ptr<ConcreteType> make_shared_op(Args&&... args) {
        return std::make_shared<ConcreteType>(std::forward<Args>(args)...);
    }

    /**
     * Returns the result of this operation
     */
    virtual xt::xarray<double> compute(xt::xarray<double> input) = 0;

    virtual std::string name() {
        return "TensorOperator";
    }

    virtual ~TensorOperator() = default;
};



class Increment : public TensorOperator {
public:
    xt::xarray<double> compute(xt::xarray<double> input) override {
        xt::xarray<double> out = input;
        for(double& i : out) {
            i += 1;
        }
        return out;
    }

    std::string name() override {
        return "increment";
    }
};


class AddFive : public TensorOperator {
public:
    xt::xarray<double> compute(xt::xarray<double> input) override {
        xt::xarray<double> out = input;
        for(double& i : out) {
            i += 5;
        }
        return out;
    }

    std::string name() override {
        return "add_five";
    }
};



}
#endif