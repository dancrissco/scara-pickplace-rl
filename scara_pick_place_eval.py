
from stable_baselines3 import PPO
from scara_pick_env_place import ScaraPickPlaceEnv
import time
import numpy as np

env = ScaraPickPlaceEnv(render=True)
model = PPO.load("./scara_rl_logs/scara_model_step_300000.zip", env=env)

successes = 0
failures = 0
total_episodes = 10

for episode in range(total_episodes):
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        time.sleep(1. / 60.)

    if env.check_success():
        successes += 1
        print(f"Episode {episode + 1}: ✅ Success")
    else:
        failures += 1
        print(f"Episode {episode + 1}: ❌ Failure")

# Final report
print("\n==== PICK AND PLACE REPORT ====")
print(f"Total Episodes:   {total_episodes}")
print(f"Successful Picks: {successes}")
print(f"Failures:         {failures}")
print(f"Success Rate:     {100 * successes / total_episodes:.1f}%")

env.close()
