# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Project Incubator** - a monorepo for experimental and proof-of-concept projects. Each project lives in its own self-contained folder under `projects/`. Projects are meant to be independent and may use different tech stacks.

## Repository Structure

```
projects/
├── _template/           # Template for new projects (copy this)
├── hello-world-example/ # Simple Python example (archived)
├── neural-elements/     # Active: Interactive neural network exploration lab
└── [your-project]/      # Create new projects here
```

## Working with Projects

Each project is self-contained. Navigate to the specific project directory and check its README.md for project-specific setup and commands.

### Creating a New Project

```bash
cp -r projects/_template projects/your-project-name
# Edit projects/your-project-name/README.md with your project details
```

### Project Status Convention

Projects use these status labels in their README: `Active`, `Paused`, `Archived`, `Promoted`

## Active Projects

### neural-elements (Python/Flask)

An interactive lab for exploring minimal neural network architectures organized like a periodic table.

**Setup:**
```bash
cd projects/neural-elements
pip install -r requirements.txt
```

**Run the web interface:**
```bash
python -m src.web.app
# Opens at http://localhost:5000
```

**Key modules:**
- `src/core/elements.py` - NeuralElement class and ElementRegistry
- `src/core/activations.py` - Activation function definitions
- `src/core/training.py` - Training loop and optimization
- `src/datasets/toy.py` - Toy datasets (XOR, spirals, moons, circles)
- `src/web/app.py` - Flask application

## Guidelines

- Projects must be self-contained (no shared code between projects)
- Each project needs a README.md with status and description
- Don't commit secrets, API keys, or large binary files
- Don't add dependencies to the repository root
