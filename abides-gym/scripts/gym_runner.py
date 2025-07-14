import gym
from tqdm import tqdm
import time

# Import to register environments
import abides_gym

if __name__ == "__main__":

    env = gym.make(
        "markets-execution-v0",
        background_config="rmsc04",
        timestep_duration="2s",
        execution_window= "01:00:00",
        mkt_close="10:30:00",
        background_config_extra_kvargs={"num_noise_agents":2000,"num_value_agents":202, "num_momentum_agents":24},
    )

    env.seed(0)
    state = env.reset()
    total_msgs =0
    reset_time = 0
    last_msgs = env.kernel.ttl_orders

    check=env.kernel.ttl_orders
    


    nsteps= 1000
    start=time.time()
    for i in tqdm(range(nsteps)):
        state, reward, done, info = env.step(0)

        new_msgs = env.kernel.ttl_orders-last_msgs
        total_msgs += new_msgs
        print(f"New orders {new_msgs}")
        avg= total_msgs / (i + 1)
        print(f"Average orders per step: {avg}")
        

        if done:
            # s_reset=time.time()
            print(f"Resetting environment at step {i+1}")
            state = env.reset()
            # e_reset=time.time()
            # reset_time += (e_reset - s_reset)
            
        last_msgs = env.kernel.ttl_orders

    end=time.time()
    # timetaken=end - start-reset_time
    print(f"Time taken for {nsteps} steps: {end - start} seconds")
    print(f"Average fps: {nsteps/(end-start)}")