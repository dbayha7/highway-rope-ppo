import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env

# Wrap the env by a RecordVideo wrapper
env = gym.make("highway-v0", render_mode="rgb_array")
env = RecordVideo(
    env, video_folder="run", episode_trigger=lambda e: True
)  # record all episodes

# Remove the incompatible set_record_video_wrapper call
# env.unwrapped.set_record_video_wrapper(env)

# Record a video as usual
obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
env.close()


def main():
    print("Hello from highway-rope-ppo!")


if __name__ == "__main__":
    main()
