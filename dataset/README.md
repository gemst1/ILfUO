# Expert Demonstration Dataset Generation
Training expert using PPO and acquire expert demonstration data. 

## Requirements
- TensorFlow higher than version 2.0
- OpenAI Gym
- MuJoCo 1.50
- mujoco-py 1.50
- RAY rllib

## Usage
- dataset.py
    - .npy dataset generation from demonstration video files.
- demon_video.py
    - generate demonstration video (.mp4) files using expert policy.
- expert_training.py
    - training expert policy using PPO.
