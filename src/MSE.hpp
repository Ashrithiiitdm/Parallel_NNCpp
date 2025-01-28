#pragma once
#include<vector>

class MSE{
    public:

        std:: vector<double> last_input;
        std:: vector<double> last_target;
        std:: vector<double> grad;

        MSE(){

        }

        ~MSE(){

        }

        double feed_forward(std:: vector<double> &inputs, std:: vector<double> &targets){
            
            this->last_input = inputs;
            this->last_target = targets;

            double sum = 0;
            int n = inputs.size();

            for(int i = 0; i < n; i++){
                double s = inputs[i] - targets[i];
                sum += s * s;
            }
            
            return sum / n;
        }


        void backward(double grad){
            this->grad = std:: vector<double>(this->last_input.size());
            int n = this->last_input.size();
            for(int i = 0; i < n; i++){
                this->grad.at(i) = 2 * (this->last_input.at(i) - this->last_target.at(i)) / this->last_input.size();
                this->grad.at(i) *= grad;
            }

        }

};