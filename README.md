# Tennis Reinforcement Learning

This project uses Unity's Machine Learning Agents (ML-Agents) to train two reinforcement learning agents to play tennis against each other. The environment is set up using Unity and Python, and the agents are trained using PPO.

## Prerequisites

To set up the environment, make sure you have the following:

- Unity (with the Tennis environment)
- Python 3.9+
- pip (Python package installer)

### Required Python Packages

1. `mlagents` for Unity environment interaction
2. `numpy` for numerical operations
3. `torch` for deep learning models
4. Other dependencies as required by the project

## Setup Instructions

### 1. Install Python 3.9 and pyenv

First, install `pyenv` if you haven't already. You can install it following the official guide: https://github.com/pyenv/pyenv

Then, install Python 3.9.13:

pyenv install 3.9.13
pyenv activate tennis-env-3.9
This will create a Python environment specifically for this project.

2. Install Dependencies
Clone the repository, navigate into the project folder, and install the necessary dependencies:


# Clone the repository (if you haven't already)
git clone https://github.com/AkshayaJeyaram/TennisReinforcementLearning.git
cd TennisReinforcementLearning

# Install dependencies using pip
pip install -r requirements.txt

Or manually install the dependencies:

pip install mlagents numpy torch

You can also install a specific version of PyTorch (1.8.1) optimized for CPU if you don't have GPU support:

pip install https://download.pytorch.org/whl/cpu/torch-1.8.1%2Bcpu-cp38-cp38-macosx_10_9_x86_64.whl

3. Set Up Unity Environment
Download the Unity Tennis environment (https://github.com/soliao/AI-Tennis-Players?tab=readme-ov-file) build and place it in the Tennis folder within the project directory. Depending on your operating system:

macOS: Tennis.app

Windows: Tennis.exe

Linux: Tennis.x86_64

4. Running the Code
Once everything is set up, you can run the training script. Make sure your environment is activated:

# Activate the Python environment
pyenv activate tennis-env-3.9

# Run the training script
python train_tennis.py
This will start the training process, where two agents will train to play tennis against each other in the Unity environment.

Additional Notes
Ensure that you have Unity installed and the ML-Agents package configured if you are planning to make changes to the Unity environment.

If you run into any issues with the Unity environment or dependencies, try to troubleshoot based on the error messages or check the Unity ML-Agents documentation: https://github.com/Unity-Technologies/ml-agents.
