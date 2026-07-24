(Forward pass completed)

NEXT MAJOR STEP: Compute a backwards pass.
- Load pre-calculation and post-calculation results in TensorOperators, during forward pass
- Verify that gradients are properly stored
- Verify Tensor and TensorOperator graph prior to backwards pass
- Compute a sequentially-defined backwards pass, and verify that weights are updated.