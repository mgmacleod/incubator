# Neural Elements - Scaling Plan

## Current State (Phase 3 Complete)

### Phase 2: Initial Exploration
**Experiments Run:** 2,316
**Coverage:**
- 8 activations: ReLU, Tanh, GELU, Sigmoid, Sine, Swish, Leaky ReLU, Linear
- 5 depths: 1-5 hidden layers
- 4 widths: 2, 4, 8, 16
- 4 datasets: XOR, moons, circles, spirals
- ~5 trials per configuration

### Phase 3: Statistical Robustness (COMPLETE)
**Experiments Run:** 2,400
**Coverage:**
- 6 activations: ReLU, Tanh, GELU, Sigmoid, Sine, Leaky ReLU
- 5 depths: 1-5 hidden layers
- Fixed width: 8
- 4 datasets: XOR, moons, circles, spirals
- 20 trials per configuration

**Key Statistical Findings:**
1. **Top activations are statistically equivalent** - Leaky ReLU, ReLU, Sine, Tanh, GELU show no significant differences (p > 0.05 after Bonferroni)
2. **Sigmoid is significantly worse** than all others (p < 0.001)
3. **Leaky ReLU vs ReLU**: 0.5% difference, NOT significant (p = 0.86)
4. **Sigmoid at depth 5**: 53.6% accuracy, approaching random chance

**Deliverables:**
- `data/phase3_summary.csv` - Aggregated statistics
- `visualizations/phase3_*.png` - Charts with confidence intervals
- `reports/phase3_statistical_summary.md` - Full report
- `examples/run_significance_tests.py` - Pairwise comparisons

---

## Phase 4: Extended Depth Study

**Goal:** Find the depth limits for each activation

**Changes:**
- Test depths 6, 7, 8, 10 layers
- Focus on sigmoid (does it hit 0%?) and sine (does it degrade?)
- Add skip connections variant (ResNet-style)

**Estimated experiments:** ~2,000
**New configurations:**
```python
# Deep networks
depths = [6, 7, 8, 10]
activations = ['sigmoid', 'relu', 'sine', 'tanh']
width = 8  # Fixed to isolate depth effect
```

**Questions to answer:**
- At what depth does sigmoid reach random chance (50%)?
- Does sine ever degrade? What's its limit?
- Do skip connections rescue sigmoid?

---

## Phase 5: Architecture Space Exploration

**Goal:** Test non-uniform architectures systematically

**New configurations:**
```python
# Bottleneck architectures
{'hidden_layers': [32, 8, 32]},     # Severe bottleneck
{'hidden_layers': [16, 4, 16]},     # Moderate bottleneck
{'hidden_layers': [8, 2, 8]},       # Extreme bottleneck

# Pyramid architectures
{'hidden_layers': [4, 8, 16]},      # Expanding
{'hidden_layers': [16, 8, 4]},      # Contracting
{'hidden_layers': [4, 8, 16, 8, 4]}, # Diamond

# Wide-shallow vs narrow-deep (same params)
{'hidden_layers': [64]},            # 1 layer, 64 wide (~200 params)
{'hidden_layers': [8, 8, 8]},       # 3 layers, 8 wide (~200 params)
```

**Estimated experiments:** ~2,500

---

## Phase 6: Learning Dynamics Study

**Goal:** Understand *how* different elements learn, not just final accuracy

**New measurements:**
- Convergence speed (epochs to 90% of final accuracy)
- Training stability (variance in loss curve)
- Gradient flow statistics
- Weight distribution evolution

**Implementation:**
```python
# Record detailed training dynamics
training_config = {
    'epochs': 1000,
    'record_every': 10,  # More frequent snapshots
    'record_gradients': True,
    'record_weight_stats': True,
}
```

---

## Phase 7: Generalization Study

**Goal:** Test if elements generalize or just memorize

**Changes:**
- Train/test split (80/20)
- Measure train vs test accuracy gap
- Test on larger datasets (n=1000+ samples)
- Add noise robustness tests

**New metrics:**
- Generalization gap = train_acc - test_acc
- Noise robustness = accuracy on noisy test data
- Sample efficiency = accuracy at 50, 100, 200 samples

---

## Phase 8: Combination & Transfer

**Goal:** Test if element properties are composable

**Experiments:**
- **Stacking:** Train element A, freeze, add element B on top
- **Transfer:** Train on dataset X, test on dataset Y
- **Ensembles:** Combine predictions from multiple elements

---

## Hardware Requirements

| Phase | Experiments | Est. Time (16-core) | Est. Time (64-core) |
|-------|-------------|---------------------|---------------------|
| 3     | 3,000       | 15 min              | 4 min               |
| 4     | 2,000       | 12 min              | 3 min               |
| 5     | 2,500       | 15 min              | 4 min               |
| 6     | 2,000       | 30 min*             | 8 min               |
| 7     | 3,000       | 20 min              | 5 min               |
| 8     | 2,000       | 15 min              | 4 min               |

*Phase 6 is slower due to additional metric recording

**Total:** ~14,500 experiments, ~2 hours on 16-core, ~30 min on 64-core

---

## Priority Order

1. **Phase 3** - Essential for publication-quality results
2. **Phase 4** - High scientific interest (depth limits)
3. **Phase 6** - Unique insights into learning dynamics
4. **Phase 5** - Architecture exploration
5. **Phase 7** - Generalization (important but slower)
6. **Phase 8** - Ambitious, good for follow-up work

---

## Quick Start Script for Phase 3

```bash
cd projects/neural-elements
python3 -c "
from examples.run_expanded_training import generate_element_configs
from src.core.persistence import ExperimentStore
from examples.train_pending import train_pending_experiments

# Generate Phase 3 configs
# ... (implementation in run_phase3.py)
"
```

---

## Notes

- All experiments are CPU-friendly (no GPU needed)
- Results are persisted in `data/experiments/` as JSON
- Visualizations auto-generate from results
- Web UI at `localhost:5000` for interactive exploration
