#ifndef Q_LEARNING_H
#define Q_LEARNING_H

#include <vector>
#include <random>
#include <algorithm>

class QLearning {
public:
    QLearning(int num_states, int num_actions, double alpha, double gamma, double epsilon);

    int selectAction(int state) const;
    void update(int state, int action, double reward, int next_state);

private:
    std::vector<std::vector<double>> q_table_;
    double alpha_;
    double gamma_;
    double epsilon_;
    int num_states_;
    int num_actions_;
    mutable std::mt19937 generator_; // Mutable to allow modification in const methods
};

#endif // Q_LEARNING_H