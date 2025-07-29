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
<video src="https://raw.githubusercontent.com/BjfpgZOC/ETH-PDM4AR-Spaceship-Project/main/out/11/index.html_resources/Evaluation-Final24-config-planet-EpisodeVisualisation-figure1-Animation.mp4" autoplay loop muted playsinline style="width:720px; height:500px; object-fit:cover; object-position:center 50%;"></video>

### Scenario 2 - Dodging a Planet and its Satellites with a Static Goal
<video src="https://raw.githubusercontent.com/BjfpgZOC/ETH-PDM4AR-Spaceship-Project/main/out/11/index.html_resources/Evaluation-Final24-config-satellites-EpisodeVisualisation-figure1-Animation.mp4" autoplay loop muted playsinline style="width:720px; height:500px; object-fit:cover; object-position:center 50%;"></video>

### Scenario 3 - Dodging a Planet and its Satellites with a Docking Goal
<video src="https://raw.githubusercontent.com/BjfpgZOC/ETH-PDM4AR-Spaceship-Project/main/out/11/index.html_resources/Evaluation-Final24-config-satellites-diff-EpisodeVisualisation-figure1-Animation.mp4" autoplay loop muted playsinline style="width:720px; height:500px; object-fit:cover; object-position:center 50%;"></video>

## How to run the code
- Clone the repository
```shell
git clone https://github.com/BjfpgZOC/ETH-PDM4AR-Spaceship-Project.git
```
- Open the folder in VS Code and **Reopen in Container**
- Run the launch configuration **"Exercise11 - Run"** (from `.vscode/launch.json`)
- Evaluations/animations are saved under `/out/11/index.html_resources/`


