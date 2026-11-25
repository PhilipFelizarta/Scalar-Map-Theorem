import ScalarMap.Activation
import ScalarMap.Order

/-!
# The Scalar Map Theorem

A multilayer perceptron `Net n m` maps `ℝⁿ` to logits in `ℝᵐ`. Every hidden layer has its own
weight matrix, bias vector and PSM activation; the last layer is affine.

* `Net.scalar_map_theorem` (Theorem 1): with no biases, scaling layer `i`'s weights by
  `c i > 0` multiplies the logits by one positive number `β`, the same for every input.
* `Net.scalar_map_theorem_bias` (Theorem 2): with biases, the same holds if layer `i`'s bias is
  scaled by the telescoping factor `c i * s_{i-1}(c (i-1) * s_{i-2}( ... c 0))`.
* `Net.topK_count_scaleW`, `Net.topK_count_scale`: hence the top-k accuracy on any finite
  dataset is unchanged, for every `k`.
* `Net.eval_scaleW_deg1`: for degree-1 activations (ReLU, leaky ReLU, ...) `β = ∏ c i`, so
  scaling every weight by `α` multiplies the logits by `α ^ depth`.
-/

noncomputable section

open Finset Matrix

namespace ScalarMap

/-- A PSM activation together with its scale function. -/
structure Act where
  g : ℝ → ℝ
  s : ℝ → ℝ
  psm : IsPSM g s

/-- A multilayer perceptron from `ℝⁿ` to `ℝᵐ`. -/
inductive Net : ℕ → ℕ → Type
  | last {n m : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (b : Fin m → ℝ) : Net n m
  | cons {n k m : ℕ} (σ : Act) (W : Matrix (Fin k) (Fin n) ℝ) (b : Fin k → ℝ)
      (rest : Net k m) : Net n m

namespace Net

/-- Number of weight matrices. -/
def depth : {n m : ℕ} → Net n m → ℕ
  | _, _, last _ _ => 1
  | _, _, cons _ _ _ r => r.depth + 1

/-- The logits (pre-softmax output). -/
def eval : {n m : ℕ} → Net n m → (Fin n → ℝ) → (Fin m → ℝ)
  | _, _, last W b, x => W *ᵥ x + b
  | _, _, cons σ W b r, x => r.eval (fun i => σ.g ((W *ᵥ x + b) i))

/-- Every bias vector is zero. -/
def NoBias : {n m : ℕ} → Net n m → Prop
  | _, _, last _ b => b = 0
  | _, _, cons _ _ b r => b = 0 ∧ r.NoBias

/-- Every hidden activation has degree 1 (`s a = a`), like ReLU and leaky ReLU. -/
def Deg1 : {n m : ℕ} → Net n m → Prop
  | _, _, last _ _ => True
  | _, _, cons σ _ _ r => (∀ a, 0 < a → σ.s a = a) ∧ r.Deg1

/-- Scale the weights of layer `i` (counted from the input, from 0) by `c i`. Biases untouched. -/
def scaleW : {n m : ℕ} → Net n m → (ℕ → ℝ) → Net n m
  | _, _, last W b, c => last (c 0 • W) b
  | _, _, cons σ W b r, c => cons σ (c 0 • W) b (r.scaleW fun i => c (i + 1))

/-- Scale layer `i`'s weights by `c i` and its bias by `c i` times the factor `a` its input has
already been scaled by. Start with `a = 1`; this is the map `φ` of Theorem 2. -/
def scale : {n m : ℕ} → Net n m → (ℕ → ℝ) → ℝ → Net n m
  | _, _, last W b, c, a => last (c 0 • W) ((c 0 * a) • b)
  | _, _, cons σ W b r, c, a =>
      cons σ (c 0 • W) ((c 0 * a) • b) (r.scale (fun i => c (i + 1)) (σ.s (c 0 * a)))

/-- The factor the logits get multiplied by: `c L * s_{L-1}( ... c 1 * s_0 (c 0 * a))`. -/
def factor : {n m : ℕ} → Net n m → (ℕ → ℝ) → ℝ → ℝ
  | _, _, last _ _, c, a => c 0 * a
  | _, _, cons σ _ _ r, c, a => r.factor (fun i => c (i + 1)) (σ.s (c 0 * a))

theorem factor_pos : ∀ {n m : ℕ} (N : Net n m) {c : ℕ → ℝ}, (∀ i, 0 < c i) →
    ∀ {a : ℝ}, 0 < a → 0 < N.factor c a
  | _, _, last _ _, _, hc, _, ha => mul_pos (hc 0) ha
  | _, _, cons σ _ _ r, _, hc, _, ha =>
      factor_pos r (fun i => hc (i + 1)) (σ.psm.scale_pos (mul_pos (hc 0) ha))

/-- The induction behind both theorems: feeding `a • x` to the scaled network gives
`factor • (logits of x)`. -/
theorem eval_scale : ∀ {n m : ℕ} (N : Net n m) {c : ℕ → ℝ}, (∀ i, 0 < c i) →
    ∀ {a : ℝ}, 0 < a → ∀ x, (N.scale c a).eval (a • x) = N.factor c a • N.eval x
  | _, _, last W b, c, hc, a, ha, x => by
      simp only [scale, eval, factor]
      rw [Matrix.smul_mulVec, Matrix.mulVec_smul, smul_smul, smul_add]
  | _, _, cons σ W b r, c, hc, a, ha, x => by
      have hpre : (c 0 • W) *ᵥ (a • x) + (c 0 * a) • b = (c 0 * a) • (W *ᵥ x + b) := by
        rw [Matrix.smul_mulVec, Matrix.mulVec_smul, smul_smul, smul_add]
      have hin : (fun i => σ.g (((c 0 • W) *ᵥ (a • x) + (c 0 * a) • b) i)) =
          σ.s (c 0 * a) • (fun i => σ.g ((W *ᵥ x + b) i)) := by
        rw [hpre]
        funext i
        simp only [Pi.smul_apply, smul_eq_mul]
        exact σ.psm.map_smul (mul_pos (hc 0) ha) _
      simp only [scale, eval, factor]
      rw [hin]
      exact eval_scale r (fun i => hc (i + 1)) (σ.psm.scale_pos (mul_pos (hc 0) ha)) _

/-- **Theorem 2 (Scalar Map Theorem with biases).** -/
theorem scalar_map_theorem_bias {n m : ℕ} (N : Net n m) {c : ℕ → ℝ} (hc : ∀ i, 0 < c i) :
    0 < N.factor c 1 ∧ ∀ x, (N.scale c 1).eval x = N.factor c 1 • N.eval x := by
  refine ⟨factor_pos N hc one_pos, fun x => ?_⟩
  have h := eval_scale N hc one_pos x
  rwa [one_smul] at h

/-- With no biases, the bias-aware scaling is just scaling the weights. -/
theorem scale_eq_scaleW : ∀ {n m : ℕ} (N : Net n m), N.NoBias → ∀ (c : ℕ → ℝ) (a : ℝ),
    N.scale c a = N.scaleW c
  | _, _, last W b, hN, c, a => by
      have hb : b = 0 := hN
      subst hb
      simp only [scale, scaleW, smul_zero]
  | _, _, cons σ W b r, hN, c, a => by
      obtain ⟨hb, hr⟩ := hN
      subst hb
      simp only [scale, scaleW, smul_zero]
      rw [scale_eq_scaleW r hr]

/-- **Theorem 1 (the Scalar Map Theorem).** In a bias-free MLP with PSM activations, scaling
each layer's weights by a positive number multiplies the logits by a single positive number
`β`, which does not depend on the input. -/
theorem scalar_map_theorem {n m : ℕ} (N : Net n m) (hN : N.NoBias) {c : ℕ → ℝ}
    (hc : ∀ i, 0 < c i) : ∃ β : ℝ, 0 < β ∧ ∀ x, (N.scaleW c).eval x = β • N.eval x := by
  refine ⟨N.factor c 1, (scalar_map_theorem_bias N hc).1, fun x => ?_⟩
  rw [← scale_eq_scaleW N hN c 1]
  exact (scalar_map_theorem_bias N hc).2 x

/-! ## Top-k accuracy -/

/-- Number of examples `d ∈ D` whose label `y d` is in the network's top `k` (so the top-k
accuracy is this divided by `D.card`). -/
def topKCount {n m : ℕ} (f : (Fin n → ℝ) → (Fin m → ℝ)) {ι : Type*} (D : Finset ι)
    (X : ι → Fin n → ℝ) (y : ι → Fin m) (k : ℕ) : ℕ :=
  haveI := Classical.decPred fun d => InTopK (f (X d)) k (y d)
  (D.filter fun d => InTopK (f (X d)) k (y d)).card

theorem topKCount_congr {n m : ℕ} {f f' : (Fin n → ℝ) → (Fin m → ℝ)} {ι : Type*}
    (D : Finset ι) (X : ι → Fin n → ℝ) (y : ι → Fin m) (k : ℕ)
    (h : ∀ x i, InTopK (f x) k i ↔ InTopK (f' x) k i) :
    topKCount f D X y k = topKCount f' D X y k := by
  classical
  unfold topKCount
  congr 1
  exact filter_congr fun d _ => h (X d) (y d)

/-- **Top-k accuracy is invariant (bias-free).** -/
theorem topK_count_scaleW {n m : ℕ} (N : Net n m) (hN : N.NoBias) {c : ℕ → ℝ}
    (hc : ∀ i, 0 < c i) {ι : Type*} (D : Finset ι) (X : ι → Fin n → ℝ) (y : ι → Fin m)
    (k : ℕ) : topKCount (N.scaleW c).eval D X y k = topKCount N.eval D X y k := by
  obtain ⟨β, hβ, h⟩ := scalar_map_theorem N hN hc
  exact topKCount_congr D X y k fun x i => by rw [h x]; exact inTopK_smul _ hβ k i

/-- **Top-k accuracy is invariant (with biases scaled as in Theorem 2).** -/
theorem topK_count_scale {n m : ℕ} (N : Net n m) {c : ℕ → ℝ} (hc : ∀ i, 0 < c i)
    {ι : Type*} (D : Finset ι) (X : ι → Fin n → ℝ) (y : ι → Fin m) (k : ℕ) :
    topKCount (N.scale c 1).eval D X y k = topKCount N.eval D X y k := by
  obtain ⟨hβ, h⟩ := scalar_map_theorem_bias N hc
  exact topKCount_congr D X y k fun x i => by rw [h x]; exact inTopK_smul _ hβ k i

/-- The same through a final softmax layer. -/
theorem topK_softmax_scale {n m : ℕ} (N : Net n m) {c : ℕ → ℝ} (hc : ∀ i, 0 < c i)
    (x : Fin n → ℝ) (k : ℕ) (i : Fin m) :
    InTopK (softmax ((N.scale c 1).eval x)) k i ↔ InTopK (softmax (N.eval x)) k i := by
  obtain ⟨hβ, h⟩ := scalar_map_theorem_bias N hc
  rw [h x]
  exact inTopK_softmax_smul _ hβ k i

/-! ## Degree-1 activations: the factor is a product -/

theorem factor_deg1 : ∀ {n m : ℕ} (N : Net n m), N.Deg1 → ∀ {c : ℕ → ℝ}, (∀ i, 0 < c i) →
    ∀ {a : ℝ}, 0 < a → N.factor c a = a * ∏ i ∈ range N.depth, c i
  | _, _, last _ _, _, c, _, a, _ => by
      simp only [factor, depth, prod_range_one]
      ring
  | _, _, cons σ _ _ r, hN, c, hc, a, ha => by
      obtain ⟨hs, hr⟩ := hN
      simp only [factor, depth]
      rw [factor_deg1 r hr (fun i => hc (i + 1)) (σ.psm.scale_pos (mul_pos (hc 0) ha)),
        hs _ (mul_pos (hc 0) ha), prod_range_succ']
      ring

/-- **Degree 1 (ReLU, leaky ReLU, PReLU, |x|).** Scaling layer `i` by `c i` multiplies the
logits by `∏ c i`. -/
theorem eval_scaleW_deg1 {n m : ℕ} (N : Net n m) (hN : N.NoBias) (hd : N.Deg1) {c : ℕ → ℝ}
    (hc : ∀ i, 0 < c i) (x : Fin n → ℝ) :
    (N.scaleW c).eval x = (∏ i ∈ range N.depth, c i) • N.eval x := by
  rw [← scale_eq_scaleW N hN c 1, (scalar_map_theorem_bias N hc).2 x,
    factor_deg1 N hd hc one_pos, one_mul]

/-- Scaling every weight by `α > 0` multiplies a degree-1 network's logits by `α ^ depth`. -/
theorem eval_scaleW_const {n m : ℕ} (N : Net n m) (hN : N.NoBias) (hd : N.Deg1) {α : ℝ}
    (hα : 0 < α) (x : Fin n → ℝ) :
    (N.scaleW fun _ => α).eval x = α ^ N.depth • N.eval x := by
  rw [eval_scaleW_deg1 N hN hd (fun _ => hα), prod_const, card_range]

end Net

end ScalarMap
