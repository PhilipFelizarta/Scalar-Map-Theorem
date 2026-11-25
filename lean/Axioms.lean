import ScalarMap
/-! Run `lake env lean Axioms.lean`: every result should depend only on
`propext`, `Classical.choice` and `Quot.sound` (no `sorryAx`). -/
#print axioms ScalarMap.Net.scalar_map_theorem
#print axioms ScalarMap.Net.scalar_map_theorem_bias
#print axioms ScalarMap.Net.topK_count_scaleW
#print axioms ScalarMap.Net.topK_count_scale
#print axioms ScalarMap.Net.eval_scaleW_deg1
#print axioms ScalarMap.Net.neuron_rescale
#print axioms ScalarMap.IsPSM.deg1_iff
#print axioms ScalarMap.IsPSM.scale_mul
#print axioms ScalarMap.inTopK_softmax_smul
