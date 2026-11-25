import Mathlib

/-!
# Top-k membership is invariant under positive scaling (Lemma 2)

`InTopK z k i` says class `i` is among the `k` largest scores of `z`: fewer than `k` classes
score strictly higher. This only looks at the order of the scores, so multiplying the logits by
a positive number, or applying softmax (which is strictly increasing), leaves it unchanged.
-/

noncomputable section

open Finset

namespace ScalarMap

/-- Class `i` is in the top `k` of the score vector `z`. -/
def InTopK {m : ℕ} (z : Fin m → ℝ) (k : ℕ) (i : Fin m) : Prop :=
  (univ.filter fun j => z i < z j).card < k

/-- Top-1 is the argmax. -/
theorem inTopK_one {m : ℕ} (z : Fin m → ℝ) (i : Fin m) : InTopK z 1 i ↔ ∀ j, z j ≤ z i := by
  unfold InTopK
  rw [Nat.lt_one_iff, card_eq_zero, filter_eq_empty_iff]
  simp [not_lt]

theorem inTopK_smul {m : ℕ} (z : Fin m → ℝ) {a : ℝ} (ha : 0 < a) (k : ℕ) (i : Fin m) :
    InTopK (a • z) k i ↔ InTopK z k i := by
  unfold InTopK
  have e : (univ.filter fun j => (a • z) i < (a • z) j) = univ.filter fun j => z i < z j :=
    filter_congr fun j _ => by
      simp only [Pi.smul_apply, smul_eq_mul]
      exact ⟨fun h => lt_of_mul_lt_mul_left h ha.le, fun h => mul_lt_mul_of_pos_left h ha⟩
  rw [e]

/-- Softmax. -/
def softmax {m : ℕ} (z : Fin m → ℝ) : Fin m → ℝ :=
  fun i => Real.exp (z i) / ∑ j, Real.exp (z j)

theorem softmax_lt_softmax {m : ℕ} (z : Fin m → ℝ) (i j : Fin m) :
    softmax z i < softmax z j ↔ z i < z j := by
  have hpos : 0 < ∑ k, Real.exp (z k) :=
    sum_pos (fun k _ => Real.exp_pos _) ⟨i, mem_univ _⟩
  unfold softmax
  rw [div_lt_div_iff_of_pos_right hpos, Real.exp_lt_exp]

theorem inTopK_softmax {m : ℕ} (z : Fin m → ℝ) (k : ℕ) (i : Fin m) :
    InTopK (softmax z) k i ↔ InTopK z k i := by
  unfold InTopK
  have e : (univ.filter fun j => softmax z i < softmax z j) = univ.filter fun j => z i < z j :=
    filter_congr fun j _ => softmax_lt_softmax z i j
  rw [e]

/-- **Lemma 2.** Softmax keeps its top-k classes when its input is scaled by `a > 0`. -/
theorem inTopK_softmax_smul {m : ℕ} (z : Fin m → ℝ) {a : ℝ} (ha : 0 < a) (k : ℕ) (i : Fin m) :
    InTopK (softmax (a • z)) k i ↔ InTopK (softmax z) k i := by
  rw [inTopK_softmax, inTopK_softmax, inTopK_smul z ha]

end ScalarMap
