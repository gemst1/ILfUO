# Imitation Learning from Unpaired Observation (ILfUO)

## Usage
1. Train expert and collect demonstration data using ["dataset"](../dataset/README.md)
2. Train the translator.
    - For example, execute ["ILfUO_pusher_sim.py"](./pusher/motion_prediction/ILfUO_pusher_sim.py)
    - To do this, you need to edit your gym environment. Please refer my ["gym"](../gym).
    - You can check learning process using Tensorboard.
        - tensorboard --logdir=./
3. Train ILfUO agent using the trained motion prediction network and RL algorithm (ex. PPO)
    - For example, execute ["train.py"](./pusher/ILfUO_rl_train/train.py)
    - To do this, you need OpenAI Baselines tf2 branch and "custom_ppo2", "custom_run.py" and "custom_cmd_util.py".
    Please refer to my [repository](https://github.com/gemst1/baselines)
    
## Results
### Motion (video) prediction results
Generate future frames from given context images.<br>

|Context Image|Generated|Ground Truth|
|:-----:|:-----:|:-----:|
|<img src="./pusher/motion_prediction/results/gif/1000_src_0.png" width="150px" height="150px">|<img src="./pusher/motion_prediction/results/gif/1000_recon_0.gif" width="150px" height="150px">|<img src="./pusher/motion_prediction/results/gif/1000_src_0.gif" width="150px" height="150px">|
|<img src="./pusher/motion_prediction/results/gif/1000_src_1.png" width="150px" height="150px">|<img src="./pusher/motion_prediction/results/gif/1000_recon_1.gif" width="150px" height="150px">|<img src="./pusher/motion_prediction/results/gif/1000_src_1.gif" width="150px" height="150px">|
|<img src="./pusher/motion_prediction/results/gif/1000_src_2.png" width="150px" height="150px">|<img src="./pusher/motion_prediction/results/gif/1000_recon_2.gif" width="150px" height="150px">|<img src="./pusher/motion_prediction/results/gif/1000_src_2.gif" width="150px" height="150px">|
|<img src="./pusher/motion_prediction/results/gif/1000_src_3.png" width="150px" height="150px">|<img src="./pusher/motion_prediction/results/gif/1000_recon_3.gif" width="150px" height="150px">|<img src="./pusher/motion_prediction/results/gif/1000_src_3.gif" width="150px" height="150px">|
|<img src="./pusher/motion_prediction/results/gif/1000_src_4.png" width="150px" height="150px">|<img src="./pusher/motion_prediction/results/gif/1000_recon_4.gif" width="150px" height="150px">|<img src="./pusher/motion_prediction/results/gif/1000_src_4.gif" width="150px" height="150px">|

