from abides_markets.configs import rmsc04
from abides_core import abides
import gym
import abides_gym
import random
import time
import logging
import coloredlogs

normal_abides = True
gym_abides = False

config_state = rmsc04.build_config(seed = 0, end_time = '16:00:00',stdout_log_level="INFO",)

if normal_abides:
    end_state = abides.run(config_state, log_dir="logs_go_here")


if gym_abides:
    env = gym.make(
    "markets-daily_investor-v0",
    background_config="rmsc04",
    )

    env.seed(0)
    initial_state = env.reset()

    num_steps = 10000
    start_time = time.time()

    logger = logging.getLogger("abides")
    coloredlogs.install(
        level=config_state["stdout_log_level"],
        fmt="[%(process)d] %(levelname)s %(name)s %(message)s",
    )
    for i in range(num_steps):
        print(f"Step {i+1}/{num_steps}")
        action = random.choice([0, 1, 2])
        state, reward, done, info = env.step(action)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for {num_steps} steps: {total_time:.4f} seconds")
    print(f"Average time per step: {total_time / num_steps:.6f} seconds")