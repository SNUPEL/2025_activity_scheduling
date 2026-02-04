# 2025_activity_scheduling

|                 Developer                |                 
| :--------------------------------------: | 
| [Hong Lee] |
|         🧑‍💻 AI-Development               |       

## Project Overview
- **Project**
    - Activity Scheduling / Resource Leveling Problem (RLP)
- **Superviser**
    - Prof. Jong Hun Woo (SNU, KOREA)
- **Data provider**
    - HD Korea Shipbuilding & Offshore Engineering (KSOE)

## Project Introduction
This project develops a hybrid scheduling algorithm designed to optimize the start times of activities. 
The primary objective is to minimize the daily resource variance, ensuring a smooth and efficient workflow.
By moving beyond traditional rule-based heuristics, we combine Pointer Networks with an Attention Mechanism to effectively navigate complex precedence relationships and time constraints.
<<img width="900" height="706" alt="image" src="https://github.com/user-attachments/assets/d28ac756-1d67-499c-8a6d-cfdce3853cee" />>


## Main Function

### 1️⃣ Overall framework
#### 1.1 Algorithm overview
<<img width="1424" height="813" alt="image" src="https://github.com/user-attachments/assets/f0984e36-48cb-422b-95bb-9f0aebfef5fe" />>

- Heuristic + Deep Reinforcement Learning: Activity sequencing is determined via enhanced Kahn's algorithm, while Activity scheduling is decided by the reinforcement agent.

### 2️⃣ Markov decision process

#### 2.1 State
- State composed of unscheduled activity feature
    - **Activity feature**: Composed of Duration, Resource requirement, number of Predecessors/Successors, and current Progress Ratio.
    - **Resource utilization**: Time-series data representing the projected daily resource utilization of the entire project for each candidate start time.

#### 2.2 Action
- Pointer-based Selection: Selecting one specific start time from the list of Valid Start Times for the current activity.
    - **Constraints**
        - Precedence relationships
        - Available time windows

#### 2.3 Reward
- The reward function is defined by the reduction in resource variance, measuring how much more effectively the model levels resources compared to heuristic.
    - **Variance**: Represents the daily fluctuation of resource usage throughout the project duration
    - **Optimization Goal**: A higher positive reward indicates that the Pointer Network has achieved a flatter resource histogram (lower variance) than the Greedy Heuristic

### 3️⃣ Scheduling environment
- Project Simulator: An engine that generates random activities and automatically configures precedence graphs
- Step-by-step Transition: Updates the global resource state in real-time as each action is taken, feeding the new state back to the agent

### 4️⃣ Scheduling Agent with Attention Network
#### 4.1 Network Structure
<<img width="1583" height="672" alt="image" src="https://github.com/user-attachments/assets/84c0915e-39ee-407a-ab3b-eb9d53a385ff" />?

- **Embedding Module**
    - Transforms raw activity features and resource profiles into high-dimensional latent vectors
- **Attention Module (Pointer Mechanism)**
    - Self-Attention: Analyzes the correlation between different candidate time slots
    - Pointer Query: Uses a learnable context vector to "point" to the most suitable start time
- **Residual Connection**
    - Combines the original feature vectors with the attention output to prevent information loss and stabilize training

#### 4.2 Reinforcement Learning Algorithm
- **Policy Gradient (REINFORCE with Baseline)**
    - Utilizes the Greedy Heuristic as a baseline to reduce variance during training and ensure efficient policy updates


