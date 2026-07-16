#include <iostream>
#include <memory>
#include <xtensor/io/xio.hpp>
#include "include/network.hpp"

int main() {
  using namespace std;
  using namespace xt;
  using namespace tasc;

  Network net;


  net.add_operator(TensorOperator::make_shared_op<Increment>());
  net.add_operator(TensorOperator::make_shared_op<TimesTwo>());
  cout << net << endl;

  for(std::shared_ptr<TensorOperator> op_ptr : net.operators_) {
    cout << *op_ptr << endl;
  }

  xarray<double> input = {1, 2, 3};
  xarray<double> output = net.forward(input);

  cout << output << endl;
  for(std::shared_ptr<TensorOperator> op_ptr : net.operators_) {
    cout << *op_ptr << endl;
  }
}

