import os
import platform
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
import numpy as np

# Automatically choose the correct file
base_path = os.path.dirname(os.path.abspath(__file__))
env_dir = os.path.join(base_path, "Tennis")

if platform.system() == "Darwin":
    env_path = os.path.join(env_dir, "Tennis.app")
elif platform.system() == "Windows":
    env_path = os.path.join(env_dir, "Tennis.exe")
else:  # Linux
    env_path = os.path.join(env_dir, "Tennis.x86_64")

env = UnityEnvironment(file_name=env_path, no_graphics=False)

env.reset()

behavior_name = list(env.behavior_specs.keys())[0]
print(f"Behavior name: {behavior_name}")

spec = env.behavior_specs[behavior_name]

for episode in range(3):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    for step in range(100):
        action = np.random.uniform(low=-1.0, high=1.0, size=spec.action_spec.continuous_size)
        action = ActionTuple(continuous=np.array([action for _ in range(len(decision_steps))]))

        env.set_actions(behavior_name, action)
        env.step()

        decision_steps, terminal_steps = env.get_steps(behavior_name)

        for agent_id in decision_steps.agent_id:
            reward = decision_steps[agent_id].reward
            print(f"Step {step} - Agent {agent_id} - Reward: {reward}")

env.close()
