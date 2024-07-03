#include "q_learning/q_learning.h"

QLearning::QLearning(int num_states, int num_actions, double alpha, double gamma, double epsilon)
    : q_table_(num_states, std::vector<double>(num_actions, 0.0)),
      alpha_(alpha),
      gamma_(gamma),
      epsilon_(epsilon),
      num_states_(num_states),
      num_actions_(num_actions),
      generator_(std::random_device{}()) {}

int QLearning::selectAction(int state) const {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    // Exploration vs. Exploitation: epsilon-greedy strategy
    if (dist(generator_) < epsilon_) {
        // Explore: choose a random action
        std::uniform_int_distribution<int> action_dist(0, num_actions_ - 1);
        return action_dist(generator_);
    }
    // Exploit: choose the action with the highest Q-value
    return std::max_element(q_table_[state].begin(), q_table_[state].end()) - q_table_[state].begin();
}

void QLearning::update(int state, int action, double reward, int next_state) {
    double best_next_action_value = *std::max_element(q_table_[next_state].begin(), q_table_[next_state].end());
    q_table_[state][action] = q_table_[state][action] + alpha_ * (reward + gamma_ * best_next_action_value - q_table_[state][action]);
}