# Tennis Reinforcement Learning

This project uses Unity's Machine Learning Agents (ML-Agents) to train two reinforcement learning agents to play tennis against each other. The environment is set up using Unity and Python, and the agents are trained using PPO and DDPG.

## Setup Instructions

# 1. Install the Anaconda Navigator
Download from https://www.anaconda.com/products/navigator, following the guidelines

# 2. Create a Python 3.6 environment
Run this command in the Anaconda Prompt 
```bash
conda create --name RL python=3.6
```

# 3. Clone this repository
(Assuming git package is installed on your device) 
```bash
git clone https://github.com/AkshayaJeyaram/TennisReinforcementLearning.git
cd TennisReinforcementLearning
pip install -r requirements.txt
```

Note: If you are on windows and you get the error "Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0)" occurs, refer to duhgrando's answer on this thread:
https://github.com/udacity/deep-reinforcement-learning/issues/13

# 4. Set Up Unity Environment

Download the Tennis Environment (Udacity's modified version) build and place it in the Tennis folder within the project directory. Depending on your operating system:

Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip

Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip

Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip

Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

(Note the PPO code was done on a windows machine, so the unity file present in the ```George_PPO``` and ```PPO-shreyas-unity-tennis``` folders is compatible with windows OS and should be replaced with the respective file as supported by your machine)

# Running the Code
Once these steps have been followed, open the respective PPO and DDPG folders and follow the instructions there on how to run the code.

Additional Notes
Ensure that you have Unity installed and the ML-Agents package configured if you are planning to make changes to the Unity environment.

