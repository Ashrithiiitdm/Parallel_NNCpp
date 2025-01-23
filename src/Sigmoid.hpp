#pragma once
#include<iostream>
#include<vector>
#include<cmath>
#include "Layer.hpp"

class Sigmoid{
    public:

        Sigmoid(){

        }

        ~Sigmoid(){

        }
        
        std::vector<double> last_input;
        std::vector<double> grad;
        std::vector<double> last_output;

        std:: vector<double> feed_forward(const std:: vector<double> &inputs){
            this->last_input = inputs;
            std:: vector<double> outputs = std:: vector<double>(inputs.size());

            for(size_t i = 0; i < inputs.size(); i++){
                outputs[i] = 1 / (1 + exp(-inputs[i]));
            }

            this->last_output = outputs;
            return outputs;
        }

        void backward(std:: vector<double> chain_grad){
            this->grad = std:: vector<double>(this->last_input.size());

            for(size_t i = 0; i < this->last_input.size(); i++){
                this->grad.at(i) = this->last_output.at(i) * (1 - this->last_output.at(i)) * chain_grad.at(i);
            }
        }

        void backward(Layer &prevlayer){
            this->grad = std:: vector<double>(this->last_input.size());

            for(size_t i = 0; i < this->last_input.size(); i++){
                double sum = 0;

                for(size_t j = 0; j < prevlayer.neurons.size(); j++){
                    sum += prevlayer.neurons[j].weights[i] * prevlayer.neurons[j].wgrad[i];
                }

                this->grad.at(i) = this->last_output.at(i) * (1 - this->last_output.at(i)) * sum;
            }
        }

};