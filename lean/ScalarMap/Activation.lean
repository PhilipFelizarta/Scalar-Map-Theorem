import Mathlib

/-!
# Positive scalar maps

A function `g : ℝ → ℝ` is a *positive scalar map* (PSM) with scale function `s` when
`g (a * x) = s a * g x` for every `a > 0`, and `s` takes positive values (Definition 1).

* Examples: ReLU, leaky ReLU, `|x|` and the identity (`s a = a`), `ReLU ^ p` (`s a = a ^ p`),
  and the Heaviside step (`s a = 1`).
* `IsPSM.scale_mul`: the scale function is multiplicative, `s (a * b) = s a * s b`,
  as soon as `g` is not identically zero. (With mild regularity this forces `s a = a ^ p`.)
* `IsPSM.deg1_iff`: the PSMs with `s a = a` are exactly the two-slope maps
  `x ↦ p * max x 0 - q * max (-x) 0`.
-/

namespace ScalarMap

/-- `g` is a positive scalar map with scale function `s`. -/
def IsPSM (g s : ℝ → ℝ) : Prop :=
  (∀ a : ℝ, 0 < a → 0 < s a) ∧ ∀ a : ℝ, 0 < a → ∀ x : ℝ, g (a * x) = s a * g x

namespace IsPSM

variable {g s : ℝ → ℝ}

theorem scale_pos (h : IsPSM g s) {a : ℝ} (ha : 0 < a) : 0 < s a := h.1 a ha

theorem map_smul (h : IsPSM g s) {a : ℝ} (ha : 0 < a) (x : ℝ) : g (a * x) = s a * g x :=
  h.2 a ha x

/-- The scale function of a PSM that is not identically zero is multiplicative. -/
theorem scale_mul (h : IsPSM g s) {x : ℝ} (hx : g x ≠ 0) {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    s (a * b) = s a * s b := by
  have h1 := h.map_smul (mul_pos ha hb) x
  have h2 : g (a * b * x) = s a * (s b * g x) := by
    rw [mul_assoc, h.map_smul ha, h.map_smul hb]
  rw [h1] at h2
  have h3 : (s (a * b) - s a * s b) * g x = 0 := by linear_combination h2
  rcases mul_eq_zero.mp h3 with h4 | h4
  · linarith
  · exact absurd h4 hx

/-- ... and so `s 1 = 1`. -/
theorem scale_one (h : IsPSM g s) {x : ℝ} (hx : g x ≠ 0) : s 1 = 1 := by
  have e := h.scale_mul hx one_pos one_pos
  rw [mul_one] at e
  have hp := h.scale_pos one_pos
  have e2 : s 1 * (s 1 - 1) = 0 := by linear_combination -e
  rcases mul_eq_zero.mp e2 with h' | h'
  · linarith
  · linarith

/-- A PSM vanishes at `0`, unless its scale function is constantly `1` (degree 0). -/
theorem map_zero (h : IsPSM g s) : g 0 = 0 ∨ ∀ a, 0 < a → s a = 1 := by
  by_cases h0 : g 0 = 0
  · exact Or.inl h0
  · refine Or.inr fun a ha => ?_
    have e := h.map_smul ha 0
    rw [mul_zero] at e
    have e2 : (s a - 1) * g 0 = 0 := by linear_combination -e
    rcases mul_eq_zero.mp e2 with h' | h'
    · linarith
    · exact absurd h' h0

/-- **Which degree-1 activations qualify.** `g` is a PSM with `s a = a` iff it is a
two-slope (generalised leaky ReLU) map. -/
theorem deg1_iff (g : ℝ → ℝ) :
    IsPSM g (fun a => a) ↔ ∃ p q : ℝ, ∀ x, g x = p * max x 0 - q * max (-x) 0 := by
  constructor
  · intro h
    have hom : ∀ c : ℝ, 0 < c → ∀ x, g (c * x) = c * g x := fun c hc x => h.map_smul hc x
    have h0 : g 0 = 0 := by
      have e := hom 2 (by norm_num) 0
      rw [mul_zero] at e
      linarith
    refine ⟨g 1, -g (-1), fun x => ?_⟩
    rcases lt_trichotomy x 0 with hx | hx | hx
    · have e := hom (-x) (by linarith) (-1)
      rw [show -x * -1 = x by ring] at e
      rw [e, max_eq_right hx.le, max_eq_left (by linarith : (0:ℝ) ≤ -x)]
      ring
    · subst hx
      simp [h0]
    · have e := hom x hx 1
      rw [mul_one] at e
      rw [e, max_eq_left hx.le, max_eq_right (by linarith : -x ≤ 0)]
      ring
  · rintro ⟨p, q, hpq⟩
    refine ⟨fun a ha => ha, fun c hc x => ?_⟩
    change g (c * x) = c * g x
    rw [hpq, hpq]
    rcases le_total x 0 with hx | hx
    · have h1 : c * x ≤ 0 := by nlinarith
      rw [max_eq_right h1, max_eq_left (by linarith : (0:ℝ) ≤ -(c * x)),
        max_eq_right hx, max_eq_left (by linarith : (0:ℝ) ≤ -x)]
      ring
    · have h1 : 0 ≤ c * x := by nlinarith
      rw [max_eq_left h1, max_eq_right (by linarith : -(c * x) ≤ 0),
        max_eq_left hx, max_eq_right (by linarith : -x ≤ 0)]
      ring

end IsPSM

/-! ## Examples -/

/-- ReLU is a PSM with `s a = a` (Lemma 1 of the paper). -/
theorem relu_isPSM : IsPSM (fun x => max x 0) (fun a => a) := by
  refine ⟨fun a ha => ha, fun a ha x => ?_⟩
  change max (a * x) 0 = a * max x 0
  rcases le_total x 0 with hx | hx
  · have h1 : a * x ≤ 0 := by nlinarith
    rw [max_eq_right h1, max_eq_right hx, mul_zero]
  · have h1 : 0 ≤ a * x := by nlinarith
    rw [max_eq_left h1, max_eq_left hx]

/-- Leaky ReLU / PReLU with any negative slope `α` (Keras `LeakyReLU(alpha)`). -/
theorem leakyReLU_isPSM (α : ℝ) :
    IsPSM (fun x => if 0 ≤ x then x else α * x) (fun a => a) := by
  refine ⟨fun a ha => ha, fun c hc x => ?_⟩
  change (if 0 ≤ c * x then c * x else α * (c * x)) = c * (if 0 ≤ x then x else α * x)
  by_cases hx : 0 ≤ x
  · have h1 : 0 ≤ c * x := mul_nonneg hc.le hx
    simp only [h1, hx, ↓reduceIte]
  · have h1 : ¬ 0 ≤ c * x := by
      intro h
      apply hx
      nlinarith
    simp only [h1, hx, ↓reduceIte]
    ring

/-- Absolute value. -/
theorem abs_isPSM : IsPSM (fun x => |x|) (fun a => a) := by
  refine ⟨fun a ha => ha, fun a ha x => ?_⟩
  change |a * x| = a * |x|
  rw [abs_mul, abs_of_pos ha]

/-- `ReLU ^ p` (for example squared ReLU) is a PSM of degree `p`, with `s a = a ^ p`. -/
theorem reluPow_isPSM (p : ℕ) : IsPSM (fun x => max x 0 ^ p) (fun a => a ^ p) := by
  refine ⟨fun a ha => pow_pos ha p, fun a ha x => ?_⟩
  change max (a * x) 0 ^ p = a ^ p * max x 0 ^ p
  rw [show max (a * x) 0 = a * max x 0 from relu_isPSM.map_smul ha x, mul_pow]

/-- The Heaviside step is a PSM of degree 0 (`s a = 1`). -/
theorem heaviside_isPSM : IsPSM (fun x => if 0 < x then (1 : ℝ) else 0) (fun _ => 1) := by
  refine ⟨fun _ _ => one_pos, fun a ha x => ?_⟩
  change (if 0 < a * x then (1 : ℝ) else 0) = 1 * (if 0 < x then (1 : ℝ) else 0)
  by_cases hx : 0 < x
  · simp [hx, mul_pos ha hx]
  · have hx' : x ≤ 0 := not_lt.mp hx
    have h1 : ¬ 0 < a * x := by
      intro h
      nlinarith
    simp [hx, h1]

end ScalarMap
