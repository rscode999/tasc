#include <xtensor/io/xio.hpp>
#include "include/network.hpp"

int main() {
  using namespace std;
  using namespace xt;
  using namespace cast;

  Network net = Network();

  net.add_operator(make_shared<Linear1d>(3, 4));

  xarray<double> input = {1, 2, 3};

  cout << net.forward(input);
}

