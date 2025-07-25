# Models Directory

This directory is for storing model checkpoints and configurations.

## Structure

- `checkpoints/` - Model weights and checkpoints
- `configs/` - Model configuration files
- `pretrained/` - Pre-trained model downloads

## Storage Guidelines

**Important**: Avoid committing large model files to git.
- Use git-lfs for models < 100MB
- For larger models, use symlinks to external storage
- Consider using model hubs (HuggingFace, etc.) for sharing

Example symlink for large models:
```bash
ln -s /path/to/model/storage models/large_models
```