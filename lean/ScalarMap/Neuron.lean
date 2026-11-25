import ScalarMap.Network

/-!
# The symmetry is bigger than per-layer scaling

For a degree-1 activation, each hidden unit can be rescaled on its own: multiply unit `j`'s
incoming weights and bias by `d j > 0` and its outgoing weights by `1 / d j`. The network
computes exactly the same function, so its loss and accuracy do not change either.
This gives one free positive parameter per hidden unit, on top of the overall logit scale
(compare Neyshabur, Salakhutdinov & Srebro, 2015).
-/

noncomputable section

open Matrix

namespace ScalarMap

namespace Net

/-- Multiply input coordinate `j` of the first layer's weights by `d j`. -/
def scaleIn : {n m : ℕ} → Net n m → (Fin n → ℝ) → Net n m
  | _, _, last W b, d => last (W * diagonal d) b
  | _, _, cons σ W b r, d => cons σ (W * diagonal d) b r

theorem diagonal_mulVec_eq {n : ℕ} (d x : Fin n → ℝ) : diagonal d *ᵥ x = d * x := by
  funext j
  simp [mulVec_diagonal]

theorem eval_scaleIn : ∀ {n m : ℕ} (N : Net n m) (d x : Fin n → ℝ),
    (N.scaleIn d).eval x = N.eval (d * x)
  | _, _, last W b, d, x => by
      simp only [scaleIn, eval]
      rw [← Matrix.mulVec_mulVec, diagonal_mulVec_eq]
  | _, _, cons σ W b r, d, x => by
      simp only [scaleIn, eval]
      rw [← Matrix.mulVec_mulVec, diagonal_mulVec_eq]

/-- **Per-neuron rescaling leaves the network unchanged.** -/
theorem neuron_rescale {n k m : ℕ} (σ : Act) (hσ : ∀ a, 0 < a → σ.s a = a)
    (W : Matrix (Fin k) (Fin n) ℝ) (b : Fin k → ℝ) (r : Net k m) (d : Fin k → ℝ)
    (hd : ∀ j, 0 < d j) (x : Fin n → ℝ) :
    (cons σ (diagonal d * W) (d * b) (r.scaleIn d⁻¹)).eval x = (cons σ W b r).eval x := by
  simp only [eval]
  rw [eval_scaleIn]
  congr 1
  funext j
  have hpre : ((diagonal d * W) *ᵥ x + d * b) j = d j * (W *ᵥ x + b) j := by
    rw [← Matrix.mulVec_mulVec, diagonal_mulVec_eq]
    simp only [Pi.add_apply, Pi.mul_apply]
    ring
  simp only [Pi.mul_apply, Pi.inv_apply]
  rw [hpre, σ.psm.map_smul (hd j), hσ _ (hd j), ← mul_assoc, inv_mul_cancel₀ (hd j).ne',
    one_mul]

end Net

end ScalarMap
