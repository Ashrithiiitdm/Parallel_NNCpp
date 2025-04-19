#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// Define constants
#define INPUT_SIZE 784    // 28x28 images
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 10
#define BATCH_SIZE 32

// Neuron structure
typedef struct {
    double *weights;
    double bias;
    double output;
    double delta;
} Neuron;

// Layer structure
typedef struct {
    int num_neurons;
    int input_size;
    Neuron *neurons;
} Layer;

// Neural Network structure
typedef struct {
    Layer input_layer;
    Layer hidden_layer;
    Layer output_layer;
} NeuralNetwork;

// MNIST Data structure
typedef struct {
    int num_images;
    int image_size;
    unsigned char **images;
    unsigned char *labels;
} MNISTData;

// Function prototypes
double sigmoid(double x);
double sigmoid_derivative(double x);
void initialize_neuron(Neuron *neuron, int input_size);
void initialize_layer(Layer *layer, int num_neurons, int input_size);
void initialize_network(NeuralNetwork *net);
void feedforward(Layer *current_layer, Layer *previous_layer);
void calculate_output_delta(Layer *output_layer, unsigned char target);
void backpropagate(Layer *current_layer, Layer *next_layer);
void update_weights(Layer *layer, Layer *previous_layer, double learning_rate);
double calculate_mse(Layer *output_layer, unsigned char target);
void train_network(NeuralNetwork *net, MNISTData *train_data, MNISTData *test_data, int rank, int size);
void test_network(NeuralNetwork *net, MNISTData *test_data, int rank, int size);
MNISTData load_mnist_images(const char *filename);
MNISTData load_mnist_labels(const char *filename);
void free_mnist_data(MNISTData *data);
int reverse_int(int i);

// Activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Initialize a single neuron
void initialize_neuron(Neuron *neuron, int input_size) {
    neuron->weights = (double *)malloc(input_size * sizeof(double));
    for (int i = 0; i < input_size; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Random between -1 and 1
    }
    neuron->bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    neuron->output = 0.0;
    neuron->delta = 0.0;
}

// Initialize a layer with neurons
void initialize_layer(Layer *layer, int num_neurons, int input_size) {
    layer->num_neurons = num_neurons;
    layer->input_size = input_size;
    layer->neurons = (Neuron *)malloc(num_neurons * sizeof(Neuron));
    for (int i = 0; i < num_neurons; i++) {
        initialize_neuron(&layer->neurons[i], input_size);
    }
}

// Initialize the neural network
void initialize_network(NeuralNetwork *net) {
    initialize_layer(&net->input_layer, INPUT_SIZE, 0); // Input layer has no weights
    initialize_layer(&net->hidden_layer, HIDDEN_SIZE, INPUT_SIZE);
    initialize_layer(&net->output_layer, OUTPUT_SIZE, HIDDEN_SIZE);
}

// Feedforward for a layer
void feedforward(Layer *current_layer, Layer *previous_layer) {
    for (int i = 0; i < current_layer->num_neurons; i++) {
        double sum = current_layer->neurons[i].bias;
        
        for (int j = 0; j < previous_layer->num_neurons; j++) {
            sum += previous_layer->neurons[j].output * current_layer->neurons[i].weights[j];
        }
        
        current_layer->neurons[i].output = sigmoid(sum);
    }
}

// Calculate delta for output layer
void calculate_output_delta(Layer *output_layer, unsigned char target) {
    for (int i = 0; i < output_layer->num_neurons; i++) {
        double target_value = (i == target) ? 1.0 : 0.0;
        double error = target_value - output_layer->neurons[i].output;
        output_layer->neurons[i].delta = error * sigmoid_derivative(output_layer->neurons[i].output);
    }
}

// Backpropagate error to previous layer
void backpropagate(Layer *current_layer, Layer *next_layer) {
    for (int i = 0; i < current_layer->num_neurons; i++) {
        double error = 0.0;
        
        for (int j = 0; j < next_layer->num_neurons; j++) {
            error += next_layer->neurons[j].delta * next_layer->neurons[j].weights[i];
        }
        
        current_layer->neurons[i].delta = error * sigmoid_derivative(current_layer->neurons[i].output);
    }
}

// Update weights based on calculated deltas
void update_weights(Layer *layer, Layer *previous_layer, double learning_rate) {
    for (int i = 0; i < layer->num_neurons; i++) {
        for (int j = 0; j < previous_layer->num_neurons; j++) {
            layer->neurons[i].weights[j] += learning_rate * layer->neurons[i].delta * previous_layer->neurons[j].output;
        }
        layer->neurons[i].bias += learning_rate * layer->neurons[i].delta;
    }
}

// Calculate mean squared error
double calculate_mse(Layer *output_layer, unsigned char target) {
    double mse = 0.0;
    for (int i = 0; i < output_layer->num_neurons; i++) {
        double target_value = (i == target) ? 1.0 : 0.0;
        double error = target_value - output_layer->neurons[i].output;
        mse += error * error;
    }
    return mse / output_layer->num_neurons;
}

// Train the neural network
void train_network(NeuralNetwork *net, MNISTData *train_data, MNISTData *test_data, int rank, int size) {
    double total_mse = 0.0;
    int correct_predictions = 0;
    int total_samples = 0;
    
    // Determine the range of data this process should handle
    int samples_per_process = train_data->num_images / size;
    int start_index = rank * samples_per_process;
    int end_index = (rank == size - 1) ? train_data->num_images : start_index + samples_per_process;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        total_mse = 0.0;
        correct_predictions = 0;
        total_samples = 0;
        
        for (int i = start_index; i < end_index; i++) {
            // Set input layer outputs
            for (int j = 0; j < INPUT_SIZE; j++) {
                net->input_layer.neurons[j].output = (double)train_data->images[i][j] / 255.0;
            }
            
            // Feedforward
            feedforward(&net->hidden_layer, &net->input_layer);
            feedforward(&net->output_layer, &net->hidden_layer);
            
            // Calculate error
            total_mse += calculate_mse(&net->output_layer, train_data->labels[i]);
            
            // Count correct predictions
            int predicted = 0;
            double max_output = net->output_layer.neurons[0].output;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (net->output_layer.neurons[j].output > max_output) {
                    max_output = net->output_layer.neurons[j].output;
                    predicted = j;
                }
            }
            if (predicted == train_data->labels[i]) {
                correct_predictions++;
            }
            total_samples++;
            
            // Backpropagation
            calculate_output_delta(&net->output_layer, train_data->labels[i]);
            backpropagate(&net->hidden_layer, &net->output_layer);
            
            // Update weights
            update_weights(&net->output_layer, &net->hidden_layer, LEARNING_RATE);
            update_weights(&net->hidden_layer, &net->input_layer, LEARNING_RATE);
        }
        
        // Reduce MSE and accuracy across all processes
        double global_mse;
        MPI_Reduce(&total_mse, &global_mse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        int global_correct, global_total;
        MPI_Reduce(&correct_predictions, &global_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_samples, &global_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            global_mse /= global_total;
            double accuracy = (double)global_correct / global_total;
            printf("Epoch %d - MSE: %.4f, Training Accuracy: %.2f%%\n", epoch + 1, global_mse, accuracy * 100);
        }
        
        // Broadcast the updated network to all processes
        for (int i = 0; i < net->hidden_layer.num_neurons; i++) {
            MPI_Bcast(net->hidden_layer.neurons[i].weights, net->hidden_layer.input_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&net->hidden_layer.neurons[i].bias, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        
        for (int i = 0; i < net->output_layer.num_neurons; i++) {
            MPI_Bcast(net->output_layer.neurons[i].weights, net->output_layer.input_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&net->output_layer.neurons[i].bias, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
    
    // Only test after all training epochs are complete
    if (rank == 0) {
        printf("\nTraining complete. Starting testing...\n");
    }
    test_network(net, test_data, rank, size);
}

// Test the neural network
void test_network(NeuralNetwork *net, MNISTData *test_data, int rank, int size) {
    int correct_predictions = 0;
    
    // Determine the range of data this process should handle
    int samples_per_process = test_data->num_images / size;
    int start_index = rank * samples_per_process;
    int end_index = (rank == size - 1) ? test_data->num_images : start_index + samples_per_process;
    
    for (int i = start_index; i < end_index; i++) {
        // Set input layer outputs
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->input_layer.neurons[j].output = (double)test_data->images[i][j] / 255.0;
        }
        
        // Feedforward
        feedforward(&net->hidden_layer, &net->input_layer);
        feedforward(&net->output_layer, &net->hidden_layer);
        
        // Count correct predictions
        int predicted = 0;
        double max_output = net->output_layer.neurons[0].output;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (net->output_layer.neurons[j].output > max_output) {
                max_output = net->output_layer.neurons[j].output;
                predicted = j;
            }
        }
        if (predicted == test_data->labels[i]) {
            correct_predictions++;
        }
    }
    
    // Reduce correct predictions across all processes
    int global_correct;
    MPI_Reduce(&correct_predictions, &global_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        double accuracy = (double)global_correct / test_data->num_images;
        printf("Test Accuracy: %.2f%%\n", accuracy * 100);
    }
}

// Helper function to reverse integer byte order
int reverse_int(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// Load MNIST images
MNISTData load_mnist_images(const char *filename) {
    MNISTData data;
    FILE *file = fopen(filename, "rb");
    
    if (file == NULL) {
        printf("Cannot open file: %s\n", filename);
        exit(1);
    }
    
    int magic_number = 0;
    int num_images = 0;
    int rows = 0;
    int cols = 0;
    
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverse_int(magic_number);
    
    fread(&num_images, sizeof(num_images), 1, file);
    num_images = reverse_int(num_images);
    
    fread(&rows, sizeof(rows), 1, file);
    rows = reverse_int(rows);
    
    fread(&cols, sizeof(cols), 1, file);
    cols = reverse_int(cols);
    
    data.num_images = num_images;
    data.image_size = rows * cols;
    data.images = (unsigned char **)malloc(num_images * sizeof(unsigned char *));
    
    for (int i = 0; i < num_images; i++) {
        data.images[i] = (unsigned char *)malloc(data.image_size * sizeof(unsigned char));
        fread(data.images[i], sizeof(unsigned char), data.image_size, file);
    }
    
    fclose(file);
    return data;
}

// Load MNIST labels
MNISTData load_mnist_labels(const char *filename) {
    MNISTData data;
    FILE *file = fopen(filename, "rb");
    
    if (file == NULL) {
        printf("Cannot open file: %s\n", filename);
        exit(1);
    }
    
    int magic_number = 0;
    int num_labels = 0;
    
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverse_int(magic_number);
    
    fread(&num_labels, sizeof(num_labels), 1, file);
    num_labels = reverse_int(num_labels);
    
    data.num_images = num_labels;
    data.image_size = 0; // Labels don't have image size
    data.labels = (unsigned char *)malloc(num_labels * sizeof(unsigned char));
    fread(data.labels, sizeof(unsigned char), num_labels, file);
    
    fclose(file);
    return data;
}

// Free MNIST data
void free_mnist_data(MNISTData *data) {
    if (data->image_size > 0) {
        for (int i = 0; i < data->num_images; i++) {
            free(data->images[i]);
        }
        free(data->images);
    }
    if (data->labels != NULL) {
        free(data->labels);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, end_time;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        start_time = MPI_Wtime();
        printf("Starting Neural Network with MPI\n");
    }
    
    // Initialize random seed
    srand(time(NULL) + rank);
    
    // Load MNIST data (only rank 0 loads the data)
    MNISTData train_data, test_data;
    
    if (rank == 0) {
        printf("Loading training data...\n");
        MNISTData train_images = load_mnist_images("dataset/train-images.idx3-ubyte");
        MNISTData train_labels = load_mnist_labels("dataset/train-labels.idx1-ubyte");
        
        printf("Loading test data...\n");
        MNISTData test_images = load_mnist_images("dataset/t10k-images.idx3-ubyte");
        MNISTData test_labels = load_mnist_labels("dataset/t10k-labels.idx1-ubyte");
        
        // Combine images and labels into single structures
        train_data.num_images = train_images.num_images;
        train_data.image_size = train_images.image_size;
        train_data.images = train_images.images;
        train_data.labels = train_labels.labels;
        
        test_data.num_images = test_images.num_images;
        test_data.image_size = test_images.image_size;
        test_data.images = test_images.images;
        test_data.labels = test_labels.labels;
    }
    
    // Broadcast the number of images to all processes
    int train_num, test_num;
    if (rank == 0) {
        train_num = train_data.num_images;
        test_num = test_data.num_images;
    }
    MPI_Bcast(&train_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Allocate memory for data on non-root processes
    if (rank != 0) {
        train_data.num_images = train_num;
        train_data.image_size = INPUT_SIZE;
        train_data.images = (unsigned char **)malloc(train_num * sizeof(unsigned char *));
        for (int i = 0; i < train_num; i++) {
            train_data.images[i] = (unsigned char *)malloc(INPUT_SIZE * sizeof(unsigned char));
        }
        train_data.labels = (unsigned char *)malloc(train_num * sizeof(unsigned char));
        
        test_data.num_images = test_num;
        test_data.image_size = INPUT_SIZE;
        test_data.images = (unsigned char **)malloc(test_num * sizeof(unsigned char *));
        for (int i = 0; i < test_num; i++) {
            test_data.images[i] = (unsigned char *)malloc(INPUT_SIZE * sizeof(unsigned char));
        }
        test_data.labels = (unsigned char *)malloc(test_num * sizeof(unsigned char));
    }
    
    // Broadcast the actual image data
    if (rank == 0) {
        for (int i = 0; i < train_num; i++) {
            MPI_Bcast(train_data.images[i], INPUT_SIZE, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        }
    } else {
        for (int i = 0; i < train_num; i++) {
            MPI_Bcast(train_data.images[i], INPUT_SIZE, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        }
    }
    
    // Broadcast labels
    MPI_Bcast(train_data.labels, train_num, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    // Broadcast test data
    if (rank == 0) {
        for (int i = 0; i < test_num; i++) {
            MPI_Bcast(test_data.images[i], INPUT_SIZE, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        }
    } else {
        for (int i = 0; i < test_num; i++) {
            MPI_Bcast(test_data.images[i], INPUT_SIZE, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Bcast(test_data.labels, test_num, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    // Initialize neural network
    NeuralNetwork net;
    initialize_network(&net);
    
    // Train the network (testing will be called from within train_network after training)
    train_network(&net, &train_data, &test_data, rank, size);
    
    // Free memory
    free_mnist_data(&train_data);
    free_mnist_data(&test_data);
    
    MPI_Finalize();

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Total time taken: %.2f seconds\n", end_time - start_time);
    }

    return 0;
}