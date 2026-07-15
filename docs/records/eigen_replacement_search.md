# Candidates for Eigen with tensors

## Criteria
- Easy to download, or come pre-compiled
- Actively maintained
- Widely used
- Easy to initialize high-dimensional tensors, with an emphasis on easy matrix and vector creation

### PRIME CANDIDATE: XTensor
Tested: 1 July 2026

Installation:
[XTensor (stable)](https://github.com/xtensor-stack/xtensor/tree/stable); [XTensor BLAS](https://github.com/xtensor-stack/xtensor-blas); [XTL 0.8.2](https://github.com/xtensor-stack/xtl/tree/master). Requires C++14 or later.

Clone each Github repo, remove the git-related directories, and use each as a header-only library.

All 3 libraries must be installed. Their "include" directories must be pointed to with the -I compiler option. Example: `-I 'lib/xtensor/include/' -I 'lib/xtl/include/' -I 'lib/xtensor-blas/include'`

Tensor initialization is notably easy.

Matrix multiplication requires the `<xtensor-blas/xlinalg.hpp>` header, done with `xt::linalg::dot(matrix1, matrix2)`. Standard multiplication is element-wise.


## Candidate List

As of 6 June 2026

- Fastor
    - Source: https://github.com/romeric/Fastor
- Autodiff
    - Source: https://github.com/autodiff/autodiff
- ATen, PyTorch's C++ interface
- C++ tensors library
    - Source: https://www.nongnu.org/tensors/doc.html
- FTensor
    - Source: https://gitlab.com/wlandry/ftensor
- TACo
    - https://github.com/tensor-compiler/taco
    - Not header-only: requires CMake build
    - Also has Python frontend
- Boost.Numeric.uBlas
    - Source: https://github.com/boostorg/ublas
    - For C++20
- ITensor
    - Still have to look at it...

## Rejected

*Boost.Numeric.uBlas*- Hard to create and initialize tensors

<details>
<summary>Notes</summary>

Can download individual source files, in theory.

Download of Boost source files resulted in all directories being empty
(even though directory structure exists).

Tensor library should exist in boost/numeric/ublas/tensor.hpp

Instructions for downloading
- Go to https://www.boost.org/releases/latest/
- Download current release for platform. Source code, not binary
    - 7z or ZIP format
    - There should be .hpp files inside
- Extract files
- Put the entire 'boost' directory inside an empty directory in the project folder. Should be {project root}/{your empty directory}/boost/...
- Compile with "-I {directory with boost/ inside}" option
    - The compiler option is an uppercase 'i'

Instructions for installing tensor library only
- Open a terminal inside Boost installation directory.
- Run these commands:
    - UNIX-like: `./bootstrap.sh; ./b2 tools/bcp`
    - Windows: `.\bootstrap.bat; .\b2 tools/bcp`
- Create new directory outside Boost installation directory
- Open terminal in Boost directory. Run the command: `./dist/bin/bcp boost/numeric/ublas/tensor.hpp {path to external directory}`

</details>
<br><br>



*ATen*- Downloaded files were too big to feasibly work with

<details>
<summary>Notes</summary>

Files were multiple GB in size.

CNet does not need ATen's autograd or gradient-storing features. Individual layers will store their own gradients.

</details>

## Resources
- https://tensornetwork.org/software/
- https://scicomp.stackexchange.com/questions/11276/fast-lightweight-c-tensor-library-for-dimension-agnostic-code