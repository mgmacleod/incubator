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

## Experimental Results (2,316 experiments)

### Key Discoveries

#### 1. Sigmoid Collapse with Depth
Sigmoid networks degrade predictably with depth due to vanishing gradients:
| Depth | Accuracy | Change |
|-------|----------|--------|
| 1 layer | 70.2% | - |
| 2 layers | 64.4% | -5.8% |
| 3 layers | 60.5% | -3.9% |
| 4 layers | 56.5% | -4.1% |
| 5 layers | **48.6%** | -7.8% |

At 5 layers, sigmoid performs **worse than random chance**.

#### 2. Sine Activation is Exceptional
- **100% XOR success** - every trial achieved perfect accuracy
- **Best on spirals** - maintains 77% accuracy at 5 layers while others degrade
- Stable training dynamics across all depths

#### 3. Activation Ranking (Overall)
| Rank | Activation | Avg Accuracy |
|------|------------|--------------|
| 1 | Leaky ReLU | 88.7% |
| 2 | Tanh | 87.2% |
| 3 | GELU | 87.2% |
| 4 | Sine | 87.1% |
| 5 | ReLU | 86.5% |
| 6 | Swish | 80.6% |
| 7 | Linear | 63.8% |
| 8 | Sigmoid | 63.7% |

#### 4. Width vs Depth Trade-off
For ReLU networks with similar parameter counts:
- `[8]` (1 layer): **90.7%**
- `[4, 4]` (2 layers): 87.9%
- `[2, 2, 2]` (3 layers): **73.0%**

**Lesson:** Width beats depth when networks are narrow. The 2-neuron bottleneck is devastating.

#### 5. Dataset-Specific Patterns
| Dataset | Best Activation | Accuracy |
|---------|-----------------|----------|
| XOR | Sine | 100% |
| Moons | Leaky ReLU | 93.6% |
| Circles | Leaky ReLU | 94.3% |
| Spirals | Sine | 72.9% |

### Visualizations

Generated charts are in `visualizations/`:
- `periodic_table_heatmap.png` - Full accuracy heatmap by depth × activation
- `depth_effect.png` - How each activation responds to depth
- `dataset_comparison.png` - Performance breakdown by dataset
- `width_effect.png` - Width vs depth trade-offs

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
