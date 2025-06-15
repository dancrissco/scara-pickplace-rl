
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from scara_pick_env_final import ScaraPickEnv

class RewardTrackerCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.rewards = []

    def _on_step(self) -> bool:
        if 'episode' in self.locals['infos'][0]:
            ep_info = self.locals['infos'][0]['episode']
            self.rewards.append(ep_info['r'])
        if self.n_calls % self.save_freq == 0:
            model_file = os.path.join(self.save_path, f"scara_model_step_{self.n_calls}.zip")
            self.model.save(model_file)
            if self.verbose > 0:
                print(f"Saved model to {model_file}")
        return True

    def save_rewards_plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Training Reward over Episodes")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "reward_curve.png"))
        plt.close()

def main():
    env = ScaraPickEnv(render=False)
    log_dir = "./scara_rl_logs"
    os.makedirs(log_dir, exist_ok=True)

    model_path = "scara_ppo_model"
    if os.path.exists(model_path + ".zip"):
        print("Loading existing model...")
        model = PPO.load(model_path, env=env, verbose=1)
    else:
        print("Starting new model...")
        model = PPO("MlpPolicy", env, verbose=1)

    callback = RewardTrackerCallback(save_freq=100_000, save_path=log_dir)
    model.learn(total_timesteps=300_000, callback=callback)
    model.save(model_path)
    callback.save_rewards_plot()

    env.close()

if __name__ == "__main__":
    main()
