# Video-o3 RL Training Guide

This guide provides detailed instructions for environment setup, data preparation, path modification, and training launch for the Video-o3 RL project.

## 1. Environment Preparation

Please follow these steps to create a Conda environment and install dependencies:

1. **Create and Activate Conda Environment**
   Python 3.11 is recommended:
   ```bash
   conda create -n video-o3_rl python=3.11 -y
   conda activate video-o3_rl
   ```

2. **Install Dependencies**
   Install the required Python packages using `requirements.txt` in the project root:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Data Preparation

Ensure that the training data is placed in the following directory:
- `annodata/RL`

**Video Data Setup:**
1. Download all required video files.
2. Extract frames from the videos at a frame rate of **4fps**.
3. Save the extracted frames to a directory of your choice.
4. Update the launch scripts (`scripts/train/train_RL_singlenodes.sh` or `scripts/train/ray_start_multinodes.sh`) by changing the `BASE_IMAGE_DIR` environment variable to your actual video frames root directory:
   ```bash
   export BASE_IMAGE_DIR="/path/to/your/video_frames"
   ```

## 3. Path Modification

Some scripts in the project may contain hardcoded old paths. Run the following command in the **project root directory** to replace all old paths with the absolute path of your current workspace:

```bash
# Replace old paths with the current directory path ($(pwd))
grep -rIl "/your_local_path_to/Video-o3/RL" . | xargs sed -i "s|/your_local_path_to/Video-o3/RL|$(pwd)|g"
```

> **Note**: This command will search for all files containing the old path in the current directory and perform an in-place replacement.

## 4. Launch Scripts

After completing the above configuration, use the following commands to start training:

### Single-Node Training
Start single-node reinforcement learning training:
```bash
bash scripts/train/train_RL_singlenodes.sh
```

### Multi-Node Training
Start Ray-based multi-node training:
```bash
bash scripts/train/ray_start_multinodes.sh
```
