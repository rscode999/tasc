#include <iostream>
#include <memory>
#include <xtensor/io/xio.hpp>
#include "include/tensor_operator.hpp"

int main() {
  using namespace std;
  using namespace xt;
  using namespace tas;

  xarray<double> input = {1, 2, 3};
  xarray<double> input2 = {100, 101, 102};
  shared_ptr<TensorNode> one_two_three = make_shared<TensorNode>(input);
  shared_ptr<TensorNode> one_hundred_etc = make_shared<TensorNode>(input2);



  shared_ptr<TensorOperator> plus_one = TensorOperator::make_shared_op<Increment>();
  shared_ptr<TensorOperator> plus_five = TensorOperator::make_shared_op<AddFive>();

  shared_ptr<TensorNode> inter1 = plus_one->compute_and_link(one_two_three);
  cout << inter1->contents() << endl; //{2, 3, 4}
  cout << inter1->prev_operator_->name() << endl; //increment
  cout << plus_one->predecessors_.at(0)->contents() << endl; //should be {1,2,3}, the original tensor
  cout << inter1->prev_operator_->predecessors_.at(0)->contents() << endl; //should be {1,2,3}, the original tensor

  cout << "------------------" << endl;

  shared_ptr<TensorNode> inter2 = plus_one->compute_and_link(one_hundred_etc);
  cout << plus_one->predecessors_.size() << endl; //should be 2
  cout << inter2->prev_operator_->predecessors_.at(1)->contents() << endl; //should be {100, 101, 102}
  
  cout << "------------------" << endl;

  shared_ptr<TensorNode> out_inter1 = plus_five->compute_and_link(inter1);
  cout << out_inter1->contents() << endl; //{7, 8, 9}
  cout << out_inter1->prev_operator_->name() << endl; //add_five
  cout << out_inter1->prev_operator_->predecessors_.at(0)->contents() << endl; //should be {2, 3, 4}, equal to inter1

  cout << "------------------" << endl;

  cout << "+5 operation's predecessors:" << endl;
  for(std::shared_ptr<TensorNode>& p : plus_five->predecessors_) {
    cout << p->contents() << endl;
  }

  cout << "increment operation's predecessors:" << endl;
  for(std::shared_ptr<TensorNode>& p : plus_one->predecessors_) {
    cout << p->contents() << endl;
  }
}

