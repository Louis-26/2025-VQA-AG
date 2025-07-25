# Data Directory

This directory contains sample data and dataset information for the TRECVID 2025 VQA Challenge.

## Structure

- `sample/` - Sample topics and videos for development
- `videos/` - Local video files for testing (create as needed)

## Video Data Storage

**Important**: Avoid storing large amounts of video data directly in this repository.
- Small sample videos are acceptable for testing
- For large datasets (MultiVent, etc.), use symlinks to external storage instead
- The official test dataset (~2000 YouTube shorts) will be distributed separately

Example symlink usage:

TODO: replace random dataset link with multivent or a more relevant data directory in the cluster
```bash
ln -s /path/to/external/dataset data/multivent
```
