#include<vector>
#include<memory>
#include "Neuron.cpp"

class Layer{
    public:

        std:: vector<Neuron> neurons;
        std:: vector<double> last_input;

        Layer(int input_size, int output_size){
            this->neurons = std:: vector<Neuron>();

            for(int i = 0; i < output_size; i++){
               Neuron n = Neuron(input_size);
               this->neurons.emplace_back(std::move(n));
            }
        }

        ~Layer(){

        }
        
        void zero_grad(){
            int n = this->neurons.size();

            for(int i = 0; i < n; i++){
                this->neurons[i].zero_grad();
            }
        }

        std:: vector<double> feed_forward(std:: vector<double> inputs){

            this->last_input = inputs;
            int n = this->neurons.size();
            std:: vector<double> outputs = std:: vector<double>(n);

            for(int i = 0; i < n; i++){
                outputs[i] = this->neurons[i].feed_forward(inputs);
            }

            return outputs;

        }

        void backward(std:: vector<double> grad){
            int n = this->neurons.size();

            for(int i = 0; i < n; i++){
                this->neurons[i].backpropagation(this->last_input, grad[i]);
            }
        }

        void descend(double learning_rate){
            int n = this->neurons.size();

            for(int i = 0; i < n; i++){
                this->neurons[i].descend(learning_rate);
            }
        }

};
