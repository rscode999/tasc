#include "tensor_operator.hpp"

#include "tensor.hpp"

#include <memory>


namespace cast {




std::shared_ptr<Tensor> TensorOperator::compute_and_link(std::shared_ptr<Tensor> input) {
    //Register the input node as a predecessor
    predecessors_.push_back(input);

    //Carry out the operation
    xt::xarray<double> output_tensor = compute(input->data());

    //Wrap operation's output in tensor node
    Tensor output_node = Tensor(output_tensor);

    //Wrap operation output tensor node in a std::shared_ptr
    std::shared_ptr<Tensor> output_node_ptr = std::make_shared<Tensor>(std::move(output_node));
    //Register this operation object as the new output's precedessor
    output_node_ptr->prev_operator_ = shared_from_this();

    return output_node_ptr;
}



std::string TensorOperator::name() {
    return "TensorOperator";
}




}