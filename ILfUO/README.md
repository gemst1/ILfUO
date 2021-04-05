# Imitation Learning from Unpaired Observation (ILfUO)

## Usage
1. Train expert and collect demonstration data using ["dataset"](../dataset/README.md)
2. Train the translator.
- For example, execute ["ILfUO_pusher_sim.py"](./pusher/motion_prediction/ILfUO_pusher_sim.py)
    - To do this, you need to edit your gym environment. Please refer my ["gym"](../gym).
    - You can check learning process using Tensorboard.
        - tensorboard --logdir=./