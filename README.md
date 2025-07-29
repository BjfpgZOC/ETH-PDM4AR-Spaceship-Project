# ETH-PDM4AR-Final-Exercise

This repository contains the final exercise for the "*Planning and Decision Making for Autonomous Robots*" course at ETH, Zurich (HS24).


## Spaceship Docking

The goal of the programming exercise was to design and implement a closed‑loop planning agent for a 2‑D spaceship that must navigate around planets and moving satellites and dock at a target while adhering to dynamics and actuator constraints. The agent reads observations at each simulation step and returns control commands.

## Constraints
- Reach the goal within position/orientation/velocity tolerances (for docking where required)
- Avoid obstacles at all times (planets/satellites)
- Respect actuator and time limits; start/end inputs are zero
- Fuel constraints: mass must remain ≥ dry mass

## Evaluation Metrics
- Mission success (goal reached safely)
- Planning efficiency (Average runtime per control step in `get_commands`)
- Time to reach the goal
- Fuel/mass consumption

## Results Summary

The planning algorithm, implemented with SCvx (Successive Convexification), was evaluated on the following 3 scenarios and animations for the performance of the algorithm are shown below.

### Scenario 1 - Dodging Planets with a Docking Goal
<image src="https://github.com/user-attachments/assets/e18f38e7-abc8-42a1-942c-3037648f0b48"></image>

### Scenario 2 - Dodging a Planet and its Satellites with a Static Goal
<image src="https://github.com/user-attachments/assets/6e7e081c-77be-4e8c-81db-a193db0ece25"></image>

### Scenario 3 - Dodging a Planet and its Satellites with a Docking Goal
<image src="https://github.com/user-attachments/assets/5a839451-a4b2-4886-bf34-967e97c6482e"></image>

## How to run the code
- Clone the repository
```shell
git clone https://github.com/BjfpgZOC/ETH-PDM4AR-Spaceship-Project.git
```
- Open the folder in VS Code and **Reopen in Container**
- Run the launch configuration **"Exercise11 - Run"** (from `.vscode/launch.json`)
- Evaluations/animations are saved under `/out/11/index.html_resources/`


