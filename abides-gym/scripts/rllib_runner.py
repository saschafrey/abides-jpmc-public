import os
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"


import ray
from ray import tune

from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback
import wandb
from abides_gym.envs.markets_execution_custom_metrics import MyCallbacks

api_key = wandb.api.api_key

# Example with custom callbacks and WandB

# Import to register environments
import abides_gym

from ray.tune.registry import register_env

# import env

from abides_gym.envs.markets_execution_environment_v0 import (
    SubGymMarketsExecutionEnv_v0,
)

register_env(
    "markets-execution-v0",
    lambda config: SubGymMarketsExecutionEnv_v0(**config),
)

ray.shutdown()

# Set RLlib to use PyTorch as the framework
import ray.rllib
ray.rllib.DEFAULT_FRAMEWORK = "torch"

"""
PPO's default:
sample_batch_size=200, batch_mode=truncate_episodes, training_batch_size=4000 -> workers collect batches of 200 ts (per worker), then the policy network gets updated using the 4000 last ts (in n minibatch-chunks of 128 (sgd_minibatch_size)).

DQN's default:
train_batch_size=32, sample_batch_size=4, timesteps_per_iteration=1000 -> workers collect chunks of 4 ts and add these to the replay buffer (of size buffer_size ts), then at each train call, at least 1000 ts are pulled altogether from the buffer (in batches of 32) to update the network.
"""
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.agents.trainer_template import build_trainer




import time
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.agents.trainer import COMMON_CONFIG


import ray
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.agents.trainer_template import build_trainer

ray.init(num_cpus=256, num_gpus=1, ignore_reinit_error=True, local_mode=False)

# for n_work in [255,127,63]:
#     # ROLLOUTS
#     RandomTrainer = build_trainer(
#     name="RandomTrainer",
#     default_policy=RandomPolicy

#     )

#     config = {"env": "markets-execution-v0",
#                 "env_config": {"timestep_duration": "2S",
#                 "mkt_close": "10:30:00",
#                 "execution_window": "01:00:00",
#                 "parent_order_size": 1000,
#                 "order_fixed_size":10,
#                 "not_enough_reward_update":-1000,
#                 "background_config_extra_kvargs":{"num_noise_agents":2000,"num_value_agents":202, "num_momentum_agents":24},
#                 },
#             "num_gpus": 0, "num_workers": n_work, "framework": None,}
#     tune.run(
#         RandomTrainer,
#         config=config,
#         stop={"timesteps_total": 10_000_000},
#     )


# TRAINING

name_xp = "ppos_execution_v1_10"
experiments = []
for n_work in [255,127]:
    for multiplier,rollout_frag in [(4,480),(1,40) , (2,240)]:
        if n_work == 255 and multiplier != 2:
            continue
        train_batch_size = rollout_frag * n_work * multiplier
        mb_size = train_batch_size // 10
        print(f"Running experiment with {n_work} workers, train_batch_size={train_batch_size}, mb_size={mb_size}, rollout_frag={rollout_frag}")
        exp=tune.run(
            "PPO",
            name=name_xp,
            resume=False,
            stop={"timesteps_total": 1_000_000},
            checkpoint_at_end=True,
            checkpoint_freq=20,
            config={
                "callbacks": None,
                "env": "markets-execution-v0",
                "env_config": {
                    "mkt_close": "10:30:00",
                    "timestep_duration": "2s",
                    "execution_window": "01:00:00",
                    "parent_order_size": 1000,
                    "order_fixed_size": tune.grid_search([10]),
                    "not_enough_reward_update": -1000,
                    "background_config_extra_kvargs":{"num_noise_agents":2000,"num_value_agents":202, "num_momentum_agents":24},
                    # "oracle_parameters": {
                    #     "l_1": tune.grid_search([-100, 0, 100]),
                    #     "sin_amp": tune.grid_search(
                    #         [
                    #             0,
                    #             100,
                    #         ]
                    #     ),
                    #     "sin_freq": tune.grid_search([2, 10, 50])
                    #     "l_2": tune.grid_search([0, -100]),
                    #     "sigma": tune.grid_search([0, 50]),
                    # },
                },
                "seed": tune.grid_search([1]),
                "num_gpus": 1,
                "num_workers": n_work,
                # "hiddens": [50],
                "gamma": 1,
                "lr": 0.0001,
                "train_batch_size":train_batch_size,
                "sgd_minibatch_size":mb_size,
                "lr_schedule": None,
                "batch_mode": "truncate_episodes",
                "rollout_fragment_length":rollout_frag,
                "framework": "torch",
                "observation_filter": "MeanStdFilter", 
            },
            callbacks=[
                WandbLoggerCallback(
                    project="abides_markets_execution_abm",
                    group=name_xp,
                    api_key=api_key,
                    log_config=False,
                )
            ],
        )
        experiments.append(exp)




ray.shutdown()
