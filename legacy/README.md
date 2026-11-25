# Legacy

The original work, kept for reference.

- `Slicing the Loss Landscape via The Scalar Invariant Theorem.ipynb`: the first experiments (Keras, MNIST).
  - It imports `layer_rotation_control` / `layer_rotation_monitoring` (Layca, Carbonnelle & De Vleeschouwer). Those modules aren't in this repo; they are in the fork `PhilipFelizarta/layer-rotation-paper-experiments` (`layca_optimizers.py`, `layer_rotation_utils.py`).
  - It picks sphere points and the scale on test data. `python/experiments/exp3_slices.py` redoes this with a validation split.
  - `np.array(model.get_weights())` builds a ragged array, which fails on NumPy ≥ 1.24.
- Earlier LaTeX drafts are not in the repo; `paper/scalar_map_theorem.tex` replaces them.
