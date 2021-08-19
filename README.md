# Double Deep Q-Learning: Intelligent investment Agent via Reinforcement Learning (TensorFlow)

<p align="center">
  <img width="460" height="300" src="https://github.com/Qu4ternion/reinforcement_learning/blob/master/img/monte_carlo.png">
</p>

## Goal:
  •   Design of an intelligent system on Python's TensorFlow that analyzes the financial data of a company's stock price in order to make autonomous investment decisions. 
   
  •   At the start the Agent takes random decisions that gradually get better as the latter gets more and more familiar with its Environment.

## Architecture:
The system is built on a Double Deep Q-Neural Network (DDQN) architecture and, as it gains more experience, its performance starts gradually improving the more the Agent explores the Environment. An "Epsilon-Greedy" approach was used to ensure that the intelligent Agent continues to explore new alternative decisions while exploiting its already learned knowledge. "Epsilon annealing" was also used to encourage the Agent to exploit the gained knowledge more often than it explores new random variations.

## Experiment:
At the beginning, the Agent is credited with an initial sum of money, which it has full freedom to decide whether to invest or not at each timestep. If the Agent invests the sum of money, the latter also decides for how long it should hold this position. At the beginning, the Agent's decisions were completely random and suboptimal, which is expected since it was merely the beginning of the process of learn this new Environment's dynamics. However, at the end of the learning loop (after reaching a threshold level of cumulative rewards), the Agent was able to beat the pre-set benchmarks and KPIs, thus outperforming the market as it became more familiar with its environment after discerning the logic of the dynamics of the evolution of the market parameters.

## Note:
Notwithstanding its success, the system can nevertheless be amply improved by introducing even more complex architectures (LSTMs, Transformers, Auto-encoders, etc.), by using more predictive input features, or simply by training it on a bigger dataset and tuning it on a distributed cluster.
