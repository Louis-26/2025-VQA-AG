# Environment Setup with uv

## Prerequisites

Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup Instructions

1. **Create and activate virtual environment:**
```bash
# Create a new virtual environment
uv venv

# Activate it (macOS/Linux)
source .venv/bin/activate

```

2. **Install dependencies:**
```bash
# Install project dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Project Structure

```
trec-project-template/
├── data/               # Place your data files here
│   ├── sample/         # Sample data for testing
│   └── videos/         # Video files (create as needed)
├── docs/               # Documentation
├── evaluation/         # Evaluation scripts (to be added)
├── examples/           # Example implementations
├── models/             # Model checkpoints and configurations
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── src/                # Source code
│   ├── ag_task/        # Answer Generation task
│   ├── mc_task/        # Multiple Choice task
│   └── utils/          # Shared utilities
├── submissions/        # Submission files
├── .gitignore          # Git ignore file
├── pyproject.toml      # Project configuration
├── README.md           # Project overview
└── SETUP.md           # This file
```

## Next Steps

1. Download the official dataset when available
2. Implement your models in the `src/` directory
3. Use the evaluation scripts to test your approach
4. Generate submissions following the format guidelines

## Common Commands

```bash
# Run linting
ruff check .

# Run type checking
mypy src/

# Run tests
pytest tests/
```
