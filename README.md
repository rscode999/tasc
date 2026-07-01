# CNet 2
**C++ framework for writing neural networks**

**Announcement: The prime candidate for CNet 2's tensor library is XTensor. See the [Eigen replacement search](docs/eigen_replacement_search.md) for more information.**  
Note that this decision is subject to change.

Proposed additions:
- different layer types, including convolutional layers
- GPU-parallelized training

Possible additions:
- use of `std::unique_ptr` to guarantee single usage