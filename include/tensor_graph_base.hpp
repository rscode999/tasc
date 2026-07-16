#ifndef TASC_TENSOR_GRAPH_
#define TASC_TENSOR_GRAPH_


#include <cstdint>
#include <memory>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>


namespace tasc {




class TensorOperator;



/**
 * Wrapper around a `xt::array` that tracks previous tensors
 */
class TensorNode {
public: //normally private

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


public:
    friend class TensorOperator;

    /**
     * Creates a new tensor node
     */
    TensorNode() = default;

    /**
     * (TEMPORARY CONSTRUCTOR) Creates a new tensor node containing `input`.
     * All other variables remain uninitialized.
     */
    TensorNode(xt::xarray<double> input) {
        data_ = input;
    }

    /**
     * @return tensor data inside the node
     */
     xt::xarray<double> data() const noexcept;

    /**
     * @return gradient of the data inside this node
     */
     xt::xarray<double> gradients() const noexcept;


    /**
     * Exports `node` to the output stream `output_stream`, returning `output_stream` with `node` inside
     * @param output_stream stream to export the node to
     * @param node tensor node to export
     * @return `output_stream` with `node`
     */
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const TensorNode& node);
};






////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/**
 * Computes an operation on one or more tensors.
 * 
 * Tracks which tensors are received and operated on by this operator.
 */
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

    /**
     * Private constructor
     */
    TensorOperator() = default;

public:


    /**
     * Returns a shared pointer to a tensor node, representing the output of this operator.
     * 
     * The node given to this method is registered as a predecessor to this operator.
     */
    std::shared_ptr<TensorNode> compute_and_link(std::shared_ptr<TensorNode> input_node);

public:

    /**
     * Makes a new shared pointer to a tensor operator.
     * 
     * Equivalent to `std::make_shared<...>()`.
     * 
     * @param args constructor arguments to the new tensor operator
     * @return `std::shared_ptr` to a new operator
     */
    template <typename ConcreteType, typename... Args>
    static std::shared_ptr<ConcreteType> make_shared_op(Args&&... args) {
        return std::make_shared<ConcreteType>(std::forward<Args>(args)...);
    }

    /**
     * Returns the result of this operation on `input`.
     * 
     * This method does not create, destroy, or modify any other objects.
     * 
     * @param input tensor to compute this operation on
     * @return result of this operator on `input`
     */
    virtual xt::xarray<double> compute(xt::xarray<double> input) = 0;

    /**
     * @return identifying string of this operator
     */
    virtual std::string name();

    /**
     * Properly destroys a tensor operator
     */
    virtual ~TensorOperator() = default;


    /**
     * Exports `op` to the output stream `output_stream`, returning `output_stream` with `op` inside
     * @param output_stream stream to export the operator to
     * @param op tensor operator to export
     * @return `output_stream` with `op` inside
     */
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const TensorOperator& op);
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



xt::xarray<double> TensorNode::data() const noexcept {
    return data_;
}



xt::xarray<double> TensorNode::gradients() const noexcept {
    return gradients_;
}



template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const TensorNode& node) {
    output_stream << "tensor node\n";
    output_stream << node.data_ << '\n';
    output_stream << "prev: " << node.prev_operator_->name() << '\n';
    return output_stream;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




std::shared_ptr<TensorNode> TensorOperator::compute_and_link(std::shared_ptr<TensorNode> input_node) {
    //Register the input node as a predecessor
    predecessors_.push_back(input_node);

    //Carry out the operation
    xt::xarray<double> output_tensor = compute(input_node->data());

    //Wrap operation's output in tensor node
    TensorNode output_node;
    output_node.data_ = output_tensor;
    output_node.gradients_ = xt::zeros<double>(output_tensor.shape());

    //Wrap operation output tensor node in a std::shared_ptr
    std::shared_ptr<TensorNode> output_node_ptr = std::make_shared<TensorNode>(std::move(output_node));
    //Register this operation object as the new output's precedessor
    output_node_ptr->prev_operator_ = shared_from_this();

    return output_node_ptr;
}



std::string TensorOperator::name() {
    return "TensorOperator";
}



template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& output_stream, const TensorOperator& op) {
    output_stream << "tensor operator\n";
    output_stream << "predecessor nodes:\n";
    for(const std::shared_ptr<TensorNode>& node_ptr : op.predecessors_) {
        output_stream << node_ptr->data_ << '\n';
    }
    output_stream << "successor nodes:\n";
    for(const std::weak_ptr<TensorNode>& node_ptr : op.successors_) {
        //lock() temporarily converts the weak pointer into a shared pointer, so it can be dereferenced
        if(std::shared_ptr<TensorNode> node_shared = node_ptr.lock()) {
            output_stream << node_shared->data_ << '\n';
        }
        //if dereferencing fails, print error
        else {
            output_stream << "(This successor node has been deallocated)" << '\n';
        }
    }
    return output_stream;
}



}
#endif