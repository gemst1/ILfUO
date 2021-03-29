import gym
from ray import tune
from ray.rllib.agents import ppo
from monitor_multi import Monitor_Multi

def every_episode(episode_id):
    return True

analysis = tune.run(ppo.PPOTrainer,
                    stop={"training_iteration":1000},
                    config={
                        "env":"Pusher3DOF-v1",
                        "monitor": True,
                        "num_workers": 4,
                        # "lambda": 0.95,
                        # "kl_coeff": 1.0,
                        # "num_sgd_iter": 16,
                        # "vf_loss_coeff": 0.5,
                        # # "sgd_minibatch_size": 4096,
                        # # "train_batch_size": 65536,
                        # "clip_param": 0.2,
                        # "grad_clip": 0.5,
                        # "lr": 0.0001,
                        # "batch_mode": "complete_episodes",
                        # "observation_filter": "MeanStdFilter"
                    },
                    local_dir='./experts',
                    checkpoint_freq=20,
                    checkpoint_at_end=True,
                    keep_checkpoints_num=10,
                    checkpoint_score_attr="episode_reward_mean"
                    )

checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean")

agent = ppo.PPOTrainer(env="Pusher3DOF-v1")
agent.restore(checkpoints[-2][0])

env = gym.make('Pusher3DOF-v1')
env = Monitor_Multi(env, './results', camera_names=["view1"], force=True)

for i in range(10):
    obs = env.reset()
    for _ in range(50):
        action = agent.compute_action(obs)
        obs, _, _, _ = env.step(action)
env.close()