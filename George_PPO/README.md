
##### George Kontis - April 2025
#### File description

```http
  ppo.py
```
Contains code for the Actor-Critic networks and the PPO agent.

```http
  mappo.py
```
Contains code for the multi agent PPO system, using previous code from ```ppo.py```.

```http
  train_evaluate.py
```
Contains code for the Generalised Advantage Estimator, RolloutBuffer, environment setup and the train and evaluate functions using ```mappo.py``` to train and test the two PPO agents.

## How to begin
Having downloaded the tennis environment for the appropriate OS, enter ```train_evaluate.py``` and specify the correct file path of the unity tennis file in the ```env``` variable.

Simply launching the terminal on this folder and entering ```python train_evaluate.py``` will start the training process for 700 episodes, generate a plot of episode rewards and evaluate the agents on 100 tennis matches.
