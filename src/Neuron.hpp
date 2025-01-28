#pragma once
#include <vector>
#include <iostream>
#include "../helpers/random.hpp"

class Neuron {
    public:
        std::vector<double> weights;
        std::vector<double> wgrad;
        double bias, bgrad;

        // Regular constructor
        Neuron(int input_size) 
            : weights(input_size, 0.0),  // Initialize weights with 0.0 by default
              bgrad(0.0)  // Initialize bgrad to 0.0
        {
            this->bias = 0.01 * get_random();

            for (int i = 0; i < input_size; i++) {
                this->weights[i] = get_random();
            }
        }

        // Copy constructor
        Neuron(const Neuron &other)
            : weights(other.weights),
              wgrad(other.wgrad),
              bias(other.bias),
              bgrad(other.bgrad)
        {
            // No need to manually reset bgrad because it's copied
        }

        // Move constructor
        Neuron(Neuron &&other) noexcept
            : weights(std::move(other.weights)), 
              wgrad(std::move(other.wgrad)),
              bias(other.bias),
              bgrad(other.bgrad)  
        {
            other.bgrad = 0.0;  // Optionally reset the bgrad of the moved-from object
        }

        ~Neuron() {
            // Destructor
        }

        void zero_grad() {
            this->wgrad = std::vector<double>(this->weights.size(), 0.0);  // Initialize wgrad with zeros
            this->bgrad = 0.0;
        }

        double feed_forward(std::vector<double> &inputs) {
            double sum = this->bias;
            int n = inputs.size();

            for (int i = 0; i < n; i++) {
                sum += inputs[i] * this->weights[i];
            }

            return sum;
        }

        void backpropagation(std::vector<double> &last_input, double grad) {
            this->bgrad += grad;

            int n = wgrad.size();

            for (int i = 0; i < n; i++) {
                this->wgrad.at(i) += grad * last_input.at(i);
            }
        }

        void descend(double learning_rate) {
            this->bias -= learning_rate * this->bgrad;

            int n = this->weights.size();

            for (int i = 0; i < n; i++) {
                this->weights.at(i) -= learning_rate * this->wgrad.at(i);
            }
        }
};
