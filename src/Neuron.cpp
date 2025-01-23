#include <vector>
#include "../helpers/print.cpp"
#include "../helpers/random.cpp"

class Neuron{
    private:

    public:
        std::vector<double> weights;
        std::vector<double> wgrad;

        double bias, bgrad;

        Neuron(int input_size){
            this->weights = std::vector<double>(input_size);
            this->bias = 0.01 * get_random();

            for (int i = 0; i < input_size; i++)
            {
                this->weights[i] = get_random();
            }
        }

        Neuron(Neuron &&other){
            this->weights = std::move(other.weights);
            this->wgrad = std::move(other.wgrad);
        }

        ~Neuron(){

        }

        void zero_grad(){
            this->wgrad = std::vector<double>(this->weights.size());
            this->bgrad = 0;
        }

        double feed_forward(std:: vector<double>inputs){
            double sum = this->bias;
            int n = inputs.size();

            for(int i = 0; i < n; i++){
                sum += inputs[i] * this->weights[i];
            }

            return sum;

        }

        void backpropagation(std:: vector<double> last_input, double grad){
            this->bgrad += grad;

            int n = wgrad.size();

            for(int i = 0; i < n; i++){
                this->wgrad.at(i) += grad * last_input.at(i);
            }
        }

        void descend(double learning_rate){
            this->bias -= learning_rate * this->bgrad;

            int n = this->weights.size();

            for(int i = 0; i < n; i++){
                this->weights.at(i) -= learning_rate * this->wgrad.at(i);
            }
            
        }


};
