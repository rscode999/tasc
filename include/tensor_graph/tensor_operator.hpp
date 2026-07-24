#ifndef CAST_TENSOR_OPERATOR_
#define CAST_TENSOR_OPERATOR_

#include <xtensor/containers/xarray.hpp>
#include <memory>
#include <string>
#include <vector>


namespace cast {




class Tensor;

/**
 * Computes an operation on one or more tensors.
 * 
 * Tracks which tensors are received and operated on by this operator.
 */
class TensorOperator : public std::enable_shared_from_this<TensorOperator> {
protected:
    /**
     * Pointers to all tensors that are computed, just prior to this operator receiving them
     */
    std::vector<std::shared_ptr<Tensor>> predecessors_ = {};


    

public:
    friend class Tensor;
    
 
    /**
     * Makes a new shared pointer to a tensor operator.
     * 
     * Equivalent to `std::make_shared<{concrete subtype of TensorOperator}>()`.
     * 
     * @param args constructor arguments to the new tensor operator
     * @return `std::shared_ptr` to a new operator
     */
    template <typename ConcreteType, typename... Args>
    static std::shared_ptr<ConcreteType> make_shared_op(Args&&... args) {
        return std::make_shared<ConcreteType>(std::forward<Args>(args)...);
    }

    /**
     * Returns a shared pointer to a tensor, representing the output of this operator.
     * The node given to this method is registered as a predecessor to this operator.
     * @param input shared pointer to the input to this operation
     * @return Tensor output of this operation
     */
    std::shared_ptr<Tensor> compute_and_link(std::shared_ptr<Tensor> input);

    /**
     * Returns the result of this operation on `input`.
     * @param input tensor to compute this operation on
     * @return result of this operator on `input`
     */
    virtual xt::xarray<double> compute(xt::xarray<double> input) const = 0;

    /**
     * Returns the result of computing a backwards pass, using this operation, on `input`.
     * @param input tensor to compute a backwards operation on
     * @return result of the backwards pass on `input`
     */
    virtual xt::xarray<double> compute_backwards_pass(xt::xarray<double> input) const = 0;

    /**
     * @return identifying string of this operator
     */
    virtual std::string name();

    /**
     * Properly destroys a tensor operator
     */
    virtual ~TensorOperator() = default;
};




}
#endif