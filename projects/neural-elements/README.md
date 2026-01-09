# Neural Elements: A Periodic Table of Neural Networks

## Status
Active

## Description

**Neural Elements** is an interactive laboratory for exploring the "atomic" building blocks of neural computation. Inspired by how the periodic table organizes elements by fundamental properties and reveals patterns, this project systematically explores *minimal* neural networks to discover:

- Which tiny configurations consistently learn specific computational primitives
- How architectural "atoms" combine to form more complex capabilities
- What patterns emerge when we organize networks by structure and behavior

### The Core Insight

Just as chemistry discovered that ~100 elements combine in predictable ways to create all matter, neural networks might have fundamental "computational atoms" - minimal configurations that reliably learn specific functions. A 3-neuron network with ReLU might always learn XOR. A specific architecture might consistently develop edge detection. By systematically exploring the space of tiny networks, we can build a "periodic table" that reveals these patterns.

### The Table Organization

- **Rows (Periods)**: Network depth (1, 2, 3+ hidden layers)
- **Columns (Groups)**: Activation families (ReLU, Tanh, Sigmoid, Sine, etc.)
- **Blocks**: Width patterns (narrow, balanced, wide)
- **Properties**: Expressiveness, training dynamics, learned representations, capacity

## Technologies

- Python 3.9+
- NumPy (core computations - no GPU required)
- Flask (web interface)
- D3.js (interactive visualizations)
- Multiprocessing (parallel bulk training)
- Optional: PyTorch (for extended experiments)

## Goals

1. **Explore** - Can we identify "neural elements" - minimal architectures with distinct computational behaviors?
2. **Organize** - Can we arrange them meaningfully, like a periodic table?
3. **Predict** - Do patterns emerge that predict behavior from structure?
4. **Synthesize** - Can we combine elements to create complex architectures?
5. **Discover** - What new insights emerge from this systematic exploration?

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
cd projects/neural-elements
pip install -r requirements.txt
```

### Running the Interactive Lab

```bash
python -m src.web.app
```

Then open http://localhost:5000 in your browser.

### Quick Start (Command Line)

```python
from src.core.elements import NeuralElement, ElementRegistry
from src.datasets.toy import spirals, xor_dataset
from src.visualization.plots import plot_decision_boundary

# Create a neural element
element = NeuralElement(
    hidden_layers=[4, 4],
    activation='relu',
    name='ReLU-2x4'
)

# Train on spiral dataset
X, y = spirals(n_samples=200)
element.fit(X, y, epochs=1000)

# Visualize
plot_decision_boundary(element, X, y)
```

## Project Structure

```
neural-elements/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── elements.py      # Neural element definitions
│   │   ├── activations.py   # Activation functions
│   │   ├── training.py      # Training loop and optimization
│   │   ├── properties.py    # Element property calculations
│   │   ├── persistence.py   # Experiment storage (JSON-based)
│   │   └── jobs.py          # Background job management
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── toy.py           # Toy datasets (XOR, spirals, etc.)
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plots.py         # Matplotlib visualizations
│   │   └── interactive.py   # D3.js data generation
│   └── web/
│       ├── __init__.py
│       ├── app.py           # Flask application
│       └── bulk_api.py      # Bulk training API endpoints
├── web/
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           ├── periodic-table.js
│           ├── training-lab.js
│           ├── experiments.js   # Bulk training UI
│           └── visualizations.js
├── data/                        # Experiment storage (auto-created)
│   ├── index.json
│   ├── jobs.json
│   └── experiments/
└── examples/
    └── explore_elements.py
```

## The Periodic Table View

The main interface presents neural elements organized like a periodic table:

```
     Linear   ReLU    Tanh    Sigmoid   Sine    GELU
    ┌──────┬───────┬───────┬─────────┬───────┬───────┐
  1 │ L-1  │ R-1   │ T-1   │  S-1    │ Si-1  │ G-1   │  ← 1 hidden layer
    ├──────┼───────┼───────┼─────────┼───────┼───────┤
  2 │ L-2  │ R-2   │ T-2   │  S-2    │ Si-2  │ G-2   │  ← 2 hidden layers
    ├──────┼───────┼───────┼─────────┼───────┼───────┤
  3 │ L-3  │ R-3   │ T-3   │  S-3    │ Si-3  │ G-3   │  ← 3 hidden layers
    └──────┴───────┴───────┴─────────┴───────┴───────┘
```

Click any element to:
- See its properties (capacity, expressiveness, training behavior)
- Train it on various toy datasets
- Visualize its learned decision boundaries
- Compare with other elements

## Features

### 1. Element Browser
Browse the periodic table of neural elements. Each element shows:
- Architecture diagram
- Key properties (parameters, capacity, typical convergence)
- Example learned functions

### 2. Training Lab
Select an element and dataset, then watch training in real-time:
- Loss curves
- Decision boundary evolution
- Gradient flow visualization
- Activation distributions

### 3. Bulk Training & Experiments
Run grid searches across multiple configurations in parallel:
- Select network depths, widths, activations, and datasets
- Training runs in background using multiprocessing
- Results stored with full weights for later analysis
- Accuracy heatmaps for comparing configurations
- Load saved experiments back into Training Lab

### 4. Synthesis Workbench
Combine multiple elements to create "molecular" architectures:
- Stack elements (sequential composition)
- Branch elements (parallel paths)
- Skip connections (residual combinations)

### 5. Discovery Mode
Systematically explore the element space:
- Run experiments across many elements
- Find patterns in what architectures learn
- Discover new "elements" with unique properties

## Current Status

- [x] Core neural element framework
- [x] Toy datasets (XOR, spirals, moons, circles)
- [x] Basic training loop
- [x] Visualization utilities
- [x] Web interface with periodic table view
- [x] Interactive training lab
- [x] Bulk training with parallel execution
- [x] Weight persistence (JSON storage)
- [x] Cross-tab experiment loading
- [ ] Synthesis workbench (in progress)
- [ ] Discovery mode automation
- [ ] Extended element library

## Bulk Training API

The bulk training system provides a REST API for running experiments programmatically:

```bash
# Submit a bulk training job
curl -X POST http://localhost:5000/api/bulk/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "elements": [
      {"hidden_layers": [4], "activation": "relu"},
      {"hidden_layers": [4, 4], "activation": "tanh"}
    ],
    "datasets": ["spirals", "xor"],
    "training": {"epochs": 500, "learning_rate": 0.1}
  }'

# Check job status
curl http://localhost:5000/api/bulk/jobs/<job_id>

# Get results with accuracy summary
curl http://localhost:5000/api/bulk/jobs/<job_id>/results

# Load a saved experiment with weights
curl http://localhost:5000/api/bulk/experiments/<experiment_id>
```

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/bulk/jobs` | POST | Submit bulk training job |
| `/api/bulk/jobs` | GET | List all jobs |
| `/api/bulk/jobs/<id>` | GET | Get job status/progress |
| `/api/bulk/jobs/<id>/results` | GET | Get results with summary |
| `/api/bulk/experiments/<id>` | GET | Load experiment with weights |
| `/api/bulk/experiments/<id>/training-lab` | GET | Get experiment in Training Lab format |

## Experimental Results

### Phase 2: Initial Exploration (2,316 experiments)

Explored 8 activations × 5 depths × 4 widths × 4 datasets with ~5 trials each.

### Phase 3: Statistical Robustness (2,400 experiments)

Focused study with 20 trials per configuration for reliable confidence intervals.

**Configuration:** 6 activations × 5 depths × 4 datasets × 20 trials

### Key Discoveries (with Statistical Validation)

#### 1. Sigmoid Collapse with Depth
Sigmoid networks degrade predictably with depth due to vanishing gradients:
| Depth | Accuracy | 95% CI | Change |
|-------|----------|--------|--------|
| 1 layer | 67.0% | [64.1%, 69.9%] | - |
| 2 layers | 64.2% | [61.2%, 67.2%] | -2.8% |
| 3 layers | 60.5% | [57.3%, 63.7%] | -3.7% |
| 4 layers | 54.4% | [51.1%, 57.7%] | -6.1% |
| 5 layers | **53.6%** | [50.2%, 57.0%] | -0.8% |

At 5 layers, sigmoid approaches **random chance (50%)**.

#### 2. Sine Activation is Exceptional
- **100% XOR success** - every trial across all depths achieved perfect accuracy
- **Best on spirals** - 83% at depth 5 with CI [80.7%, 85.3%]
- Stable training dynamics across all depths

#### 3. Activation Ranking (with 95% CI)
| Rank | Activation | Avg Accuracy | 95% CI |
|------|------------|--------------|--------|
| 1 | Leaky ReLU | 92.2% | [91.3%, 93.1%] |
| 2 | ReLU | 91.7% | [90.6%, 92.8%] |
| 3 | Sine | 90.7% | [88.8%, 92.6%] |
| 4 | Tanh | 89.8% | [88.6%, 90.9%] |
| 5 | GELU | 86.4% | [83.9%, 88.9%] |
| 6 | Sigmoid | 59.9% | [57.0%, 62.9%] |

**Statistical note:** Top 5 activations form a statistically equivalent group (p > 0.05 after Bonferroni correction). Only sigmoid is significantly worse than all others.

#### 4. Width vs Depth Trade-off
For ReLU networks with similar parameter counts:
- `[8]` (1 layer): **90.7%**
- `[4, 4]` (2 layers): 87.9%
- `[2, 2, 2]` (3 layers): **73.0%**

**Lesson:** Width beats depth when networks are narrow. The 2-neuron bottleneck is devastating.

#### 5. Dataset-Specific Patterns
| Dataset | Best Activation | Depth | Accuracy | 95% CI |
|---------|-----------------|-------|----------|--------|
| XOR | GELU | 1 | 100.0% | [100%, 100%] |
| Moons | Tanh | 5 | 100.0% | [100%, 100%] |
| Circles | Leaky ReLU | 4 | 100.0% | [100%, 100%] |
| Spirals | Sine | 5 | 83.0% | [80.7%, 85.3%] |

### Statistical Findings (Phase 3)

Using pairwise t-tests with Bonferroni correction:

1. **Sigmoid is significantly worse** than all other activations (p < 0.001)
2. **Top activations are statistically equivalent** - Leaky ReLU, ReLU, Sine, Tanh, GELU show no significant differences
3. **Leaky ReLU vs ReLU**: Only 0.5% difference, NOT significant (p = 0.86)
4. **Sine vs ReLU on spirals**: 3.4% difference, NOT significant after correction (p = 0.39)

### Phase 4: Extended Depth Study (2,560 experiments)

Tested extreme depths (6, 7, 8, 10 layers) with and without ResNet-style skip connections.

**Configuration:** 4 activations × 4 depths × 4 datasets × 20 trials × 2 variants (baseline + skip)

#### 1. Sigmoid Collapse Confirmed

Deep sigmoid networks completely collapse to random chance:

| Depth | Accuracy | Notes |
|-------|----------|-------|
| 6 | 49.8% | Random chance |
| 7 | 50.5% | Random chance |
| 8 | 49.4% | Random chance |
| 10 | 52.0% | Random chance |

The loss converges to exactly 0.693 (ln(2)), confirming complete gradient death.

#### 2. Skip Connections Rescue Sigmoid

With residual connections, sigmoid improved dramatically:

| Depth | Baseline | With Skip | Improvement |
|-------|----------|-----------|-------------|
| 6 | 50% | 76% | **+26%** |
| 7 | 51% | 73% | **+23%** |
| 8 | 49% | 75% | **+25%** |
| 10 | 52% | 72% | **+20%** |

The gradient can flow through the skip path even when the transformation path saturates.

#### 3. Sine Remains Stable at Extreme Depths

Sine activation maintained ~96% accuracy even at depth 10:
- Depth 6: 96.4%
- Depth 7: 97.2%
- Depth 8: 97.2%
- Depth 10: 96.4%

This confirms sine's exceptional gradient properties - it doesn't suffer from vanishing gradients.

#### 4. Unexpected: Skip Connections Hurt Good Activations

| Activation | Baseline (d=10) | With Skip (d=10) | Change |
|------------|-----------------|------------------|--------|
| ReLU | 93% | 50% | **-43%** |
| Sine | 96% | 50% | **-46%** |
| Tanh | 93% | 88% | -5% |
| Sigmoid | 52% | 72% | **+20%** |

Skip connections collapsed ReLU and Sine to random chance at depth 10, while rescuing sigmoid. This suggests:
- Skip connections can interfere with activations that already have good gradient flow
- The residual path may dominate and bypass the learned transformations
- Skip connections are a *targeted intervention* for vanishing gradients, not a universal improvement

#### Key Takeaways

1. **Sigmoid has a hard depth limit of ~5 layers** without architectural modifications
2. **Skip connections are activation-dependent** - they help sigmoid but hurt ReLU/Sine
3. **Sine is remarkably stable** - maintains 96%+ accuracy from depth 1 to depth 10
4. **Tanh degrades gracefully** - slight decline from 94% (d=6) to 93% (d=10)

### Phase 5: Architecture Space Exploration (3,520 experiments)

Tested non-uniform architectures including bottlenecks, pyramids, and parameter-matched configurations.

**Configuration:** 11 architectures × 4 activations × 4 datasets × 20 trials

#### 1. Bottleneck Architecture Survival

| Architecture | Pattern | Accuracy | Notes |
|--------------|---------|----------|-------|
| `[32, 8, 32]` | Severe bottleneck | 93.9% | 8-neuron bottleneck works |
| `[16, 4, 16]` | Moderate bottleneck | 92.0% | 4-neuron bottleneck works |
| `[8, 2, 8]` | Extreme bottleneck | 85.6% | Width-2 degrades (~7% loss) |

**Key insight:** Bottleneck width of 4+ neurons is sufficient. Width-2 bottlenecks degrade but don't collapse completely (unlike uniform width-2 networks).

#### 2. Pyramid Direction Doesn't Matter

| Pattern | Architecture | Accuracy |
|---------|--------------|----------|
| Expanding | `[4, 8, 16]` | 92.0% |
| Contracting | `[16, 8, 4]` | 92.0% |
| Diamond | `[4, 8, 16, 8, 4]` | 91.2% |

All pyramid patterns perform identically - **direction has no effect** on these toy datasets.

#### 3. Parameter Efficiency: Depth > Width

For similar parameter budgets (~150-200 params):

| Architecture | Params | Accuracy |
|--------------|--------|----------|
| Wide-shallow `[32]` | ~97 | 84.4% |
| Medium `[12, 12]` | ~193 | **89.6%** |
| Narrow-deep `[8, 8, 8]` | ~169 | 85.6% |

**Key insight:** Adding layers beats adding width for similar parameter counts. But the medium balanced configuration wins overall.

#### 4. Activation × Architecture Interactions

The architecture heatmap reveals:
- **ReLU/Leaky ReLU**: 92-96% across all architectures
- **Sine**: Stable ~92% but drops to 78% on wide-shallow `[32]`
- **Sigmoid**: ~61% failure across all depth-3+ architectures
- **Wide-shallow hurts periodic activations**: Sine drops 14% on `[32]` vs other architectures

#### Key Takeaways

1. **Width-2 is the critical bottleneck threshold** - confirmed across bottleneck and uniform architectures
2. **Pyramid direction is irrelevant** - expanding, contracting, diamond all equivalent
3. **Medium balanced architectures are optimal** for parameter efficiency
4. **Wide-shallow networks hurt Sine** - periodic activations need depth to express their oscillations

### Visualizations

Generated charts are in `visualizations/`:

**Phase 2:**
- `periodic_table_heatmap.png` - Full accuracy heatmap by depth × activation
- `depth_effect.png` - How each activation responds to depth
- `dataset_comparison.png` - Performance breakdown by dataset
- `width_effect.png` - Width vs depth trade-offs

**Phase 3 (with confidence intervals):**
- `phase3_depth_effect_ci.png` - Depth effect with shaded CI bands
- `phase3_activation_ranking_ci.png` - Ranking with error bars
- `phase3_dataset_comparison_ci.png` - Per-dataset with CI
- `phase3_variance_reliability.png` - Accuracy vs reliability scatter
- `phase3_activation_boxplots.png` - Distribution box plots

**Phase 4 (extended depth + skip connections):**
- `phase4_sigmoid_collapse.png` - Sigmoid baseline vs skip at depths 6-10
- `phase4_sine_stability.png` - All activations at extreme depths
- `phase4_skip_rescue.png` - Skip connection effect by activation (2x2 grid)
- `phase4_combined_depths.png` - Full depth range (Phase 3 + 4 combined)
- `phase4_depth_collapse.png` - Extended depth degradation curves

**Phase 5 (architecture space exploration):**
- `phase5_bottleneck_survival.png` - Which bottleneck widths can still learn
- `phase5_pyramid_comparison.png` - Expanding vs contracting vs diamond
- `phase5_parameter_efficiency.png` - Accuracy per parameter count
- `phase5_architecture_heatmap.png` - Pattern × activation performance grid
- `phase5_dataset_comparison.png` - Best architecture by dataset

**Reports:**
- `reports/phase3_statistical_summary.md` - Full statistical report

## Notes and Learnings

### What the Periodic Table Reveals
1. **Columns (activations) define capability families** - sigmoid/linear can't solve nonlinear tasks
2. **Rows (depth) show scaling behavior** - some activations benefit, others collapse
3. **Blocks (width) set minimum viable capacity** - below width=4, everything struggles

### Unexpected Findings
- **Swish underperformed** - expected GELU-like behavior but was much worse
- **Sine is the dark horse** - not commonly used but exceptional on difficult tasks
- **Leaky ReLU > ReLU** - the small leak makes a meaningful difference

## Future Ideas

- **Genetic Elements**: Encode architectures as "DNA" strings, evolve new elements
- **Transfer Properties**: Which elements transfer knowledge well?
- **Scaling Laws**: How do element properties change with scale?
- **Interpretability**: What computations do minimal elements actually learn?
- **Phase Transitions**: At what size do new capabilities emerge?

## Philosophy

This project embraces the idea that **understanding small things deeply reveals patterns that apply broadly**. By studying the simplest neural networks thoroughly, we might discover organizing principles that illuminate the behavior of much larger models.

The periodic table of chemistry was revolutionary not because it cataloged known elements, but because it revealed deep patterns and predicted the unknown. Perhaps a similar systematic study of neural architectures can do the same for deep learning.

## References

- Olah et al., "Neural Networks, Manifolds, and Topology" - https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
- The original periodic table story - patterns from systematic observation
- "Lottery Ticket Hypothesis" - small networks contain the essence of large ones
