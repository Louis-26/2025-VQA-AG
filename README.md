# TRECVID Project Template

## Overview

The Video Question Answering (VQA) Challenge aims to rigorously assess the capabilities of state-of-the-art multimodal models in understanding and reasoning about video content. This repository contains resources, evaluation tools, and submission guidelines for the TRECVID 2025 VQA track.

**Official Task Details**: https://www-nlpir.nist.gov/projects/tv2025/vqa.html

## Challenge Tasks

### 1. Answer Generation (AG) Task


### 2. Multiple Choice (MC) Task


## Quick Start

```bash
# Clone the repository
git clone git@github.com:debashishc/trec-project-template.git
cd trec-project-template

# Set up environment with uv
uv venv
source .venv/bin/activate
uv pip install -e .

# Run example evaluation (once evaluate pipeline is set up)
# python evaluation/evaluate.py --task ag --submission path/to/submission.csv
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## Repository Structure

```
├── data/               # Sample data and dataset information
├── docs/               # Detailed documentation
├── evaluation/         # Evaluation scripts and metrics
├── examples/           # Example code and baselines
├── models/             # Model checkpoints and configurations
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── src/                # Source code
│   ├── ag_task/        # Answer Generation task code
│   ├── mc_task/        # Multiple Choice task code
│   └── utils/          # Shared utilities
└── submissions/        # Submission format examples
```

## Dataset

- **Test Dataset**: ~2000 YouTube shorts (approximately 30 seconds each)
- **Development Data**: Teams can use publicly available VQA datasets
- Distribution details will be announced via the participants mailing list (TODO: need to follow up how to get added to the mailing list)

## Submission Guidelines

### Answer Generation Format
```csv
Q_ID, Video_ID, Rank, Answer, Time (sec)
1, tui89Xr_iri, 1, she found a surprise birthday party, 5
1, tui89Xr_iri, 2, she found party, 6
1, tui89Xr_iri, 3, she found a group of people, 8
...
1, tui89Xr_iri, 10, a dog barked at her, 10
```

### Multiple Choice Format
```csv
Q_ID, Video_ID, Rank, option_X
1, tui89Xr_iri, 1, the room was empty
1, tui89Xr_iri, 2, a dog jumped on her
1, tui89Xr_iri, 3, found a party
1, tui89Xr_iri, 4, a man surprised her
```

## Evaluation Metrics

- **Answer Generation**: STS, METEOR, BERTScore, NDCG
- **Multiple Choice**: Top-1 Accuracy, Mean Reciprocal Rank (MRR)

## Important Dates

*To be announced*

## Resources

- [Submission Instructions](TBA)
- [Answer Generation Task Details](docs/answer_generation.md)
- [Multiple Choice Task Details](docs/multiple_choice.md)

## Contact

For questions and updates, please join the active participants mailing list.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
