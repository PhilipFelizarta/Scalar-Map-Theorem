"""Numerical companion to the Lean proofs of the Scalar Map Theorem."""
from .mlp import (  # noqa: F401
    ACTIVATIONS, MLP, best_logit_scale, cross_entropy, leaky_relu, predicted_factor,
    rescale_neurons, scale_weights, scale_with_bias, softmax, topk_accuracy, train,
)
from .data import digits_splits  # noqa: F401
