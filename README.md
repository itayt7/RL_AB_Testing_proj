# RL-Based A/B Testing for E-commerce
Overview
This repository contains the implementation of a reinforcement learning (RL) based A/B testing framework for optimizing discount strategies in an e-commerce setting. The goal of this project is to use an RL agent to dynamically determine the optimal discount to offer new users to maximize their deposit amounts and improve user retention.

Table of Contents
Introduction
Concepts
A/B Testing
Reinforcement Learning
Our Approach
Dataset
Methods
Results
Conclusions
Future Work
Usage
Installation
Running the Code

Introduction
Traditional A/B testing methods are often limited by static decision-making and lack adaptability to dynamic environments. By leveraging RL, we aim to enhance A/B testing through continuous learning and adaptive decision-making to optimize long-term outcomes.

Concepts
A/B Testing
A/B testing is a method to compare two versions of a product to determine which one performs better. This project addresses the challenges in traditional A/B testing, such as managing dynamic environments and accounting for long-term effects.

Reinforcement Learning
Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions and receiving rewards. The agent in this project uses Q-learning to optimize discount strategies for new users.

Our Approach
Dataset
The dataset includes user features such as age, IP location, browsing time, pages visited, referral source, onboarding time, KYC type, browser type, payment method, time of day, planned monthly deposit, device, device model, and initial suggested deposit amount.

Methods
Modeling the Environment: We modeled the e-commerce onboarding page as the environment.
Defining States, Actions, and Rewards:
States: User feature vectors.
Actions: Offering different discount percentages (5%, 10%, 15%).
Rewards: Amount deposited by the user, with penalties for skipping or abandoning the process.
Training the Agent: The agent uses Q-learning to learn the optimal discount strategy. Q-values are updated based on the Bellman equation, balancing exploration and exploitation.
Results
The RL agent demonstrated improved performance in terms of user deposits and retention:

Increase in approval rates by approximately 8%.
Decrease in churn rates by approximately 9%.
Conclusions
Our study shows that reinforcement learning provides a powerful framework for advanced A/B testing, allowing for continuous learning and optimization of user interactions.

Future Work
Future enhancements could include:

Implementing the agent at different checkpoints in the user journey.
Collecting more user data during onboarding.
Stabilizing the training process by tuning hyperparameters.
Exploring deep learning approaches, such as Deep Q-Networks (DQN), for potentially better results.
Usage
Installation
Clone this repository and install the required dependencies:

bash
Copy code
git clone https://github.com/yourusername/RL-AB-Testing.git
cd RL-AB-Testing
pip install -r requirements.txt
Running the Code
Train the RL Agent:
bash
Copy code
python q_learning_agent.py
Run the Q-Learning Model:
bash
Copy code
python RL_QLearning_model.py
Evaluate with CatBoost Classifier:
bash
Copy code
python Catboost_classifier_RL_discount.py
References
Wang, R., & Blei, D. M. (2019). Dynamic Causal Effects Evaluation in A/B Testing with a Reinforcement Learning Framework. arXiv:2002.01711
