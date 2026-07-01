#include <iostream>
#include "include/layer.hpp"


int main() {
  using namespace std;
  using namespace xt;
  using namespace CNet;


  Layer l(3, 2);

  cout << l.weights << endl;
  cout << l.biases << endl;
  cout << l.weights << endl;

  xarray<double> input({1, 2, 3});

  cout << l.forward(input) << endl;
}

