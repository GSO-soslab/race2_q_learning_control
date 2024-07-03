# q-learning

The `q_learning` package provides a Q-learning implementation for use in ROS (Robot Operating System) environments. This package is designed to be configurable via a `.yaml` file, allowing easy tuning of parameters.

## Features

- Q-learning implementation with configurable parameters
- Integration with ROS parameter server
- Example configuration file

## Installation

1. **Clone the repository**:

    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/farhangnaderi/q-learning
    ```

2. **Build the package**:

    ```bash
    cd ~/catkin_ws
    catkin build
    source devel/setup.bash
    ```

## Configuration

The Q-learning parameters can be configured via a YAML file. The default configuration file is located in the `config` directory of this package.

### Default Configuration File

`config/config.yaml`:

```yaml
q_learning:
  num_states: 100
  num_actions: 2
  alpha: 0.1
  gamma: 0.9
  epsilon: 0.1
```

### Parameter Description

- **num_states**: The number of states in the Q-learning model.
- **num_actions**: The number of actions in the Q-learning model (e.g., thrust positive, thrust negative).
- **alpha**: The learning rate.
- **gamma**: The discount factor.
- **epsilon**: The exploration rate.

## Usage

To use the `q_learning` package in your ROS project, follow these steps:

1. **Include the Q-learning header in your code**:

    ```cpp
    #include "q_learning/q_learning.h"
    ```

2. **Load the Q-learning parameters**:

    ```cpp
    ros::NodeHandle nh;
    int num_states, num_actions;
    double alpha, gamma, epsilon;
    nh.getParam("q_learning/num_states", num_states);
    nh.getParam("q_learning/num_actions", num_actions);
    nh.getParam("q_learning/alpha", alpha);
    nh.getParam("q_learning/gamma", gamma);
    nh.getParam("q_learning/epsilon", epsilon);

    QLearning q_learning(num_states, num_actions, alpha, gamma, epsilon);
    ```

3. **Use the Q-learning instance in your control logic**:

    ```cpp
    int current_state = getCurrentState();
    int action = q_learning.selectAction(current_state);
    double thrust_direction = (action == 0) ? 1.0 : -1.0;
    // Apply thrust_direction in your control logic
    ```

4. **Update the Q-learning model**:

    ```cpp
    int next_state = getNextState();
    double reward = calculateReward();
    q_learning.update(current_state, action, reward, next_state);
    ```

### Example Launch File

An example launch file to load the default configuration and start your node:

`launch/q_learning_example.launch`:

```xml
<launch>
  <rosparam file="$(find q_learning)/config/config.yaml" command="load"/>
  <node pkg="your_package" type="your_node" name="your_node" output="screen"/>
</launch>
```

## Contact

For questions or comments, please contact [Farhang Naderi](mailto:farhang.nba@gmail.com).