%%writefile nn.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// MNIST dataset dimensions
#define MNIST_IMAGE_SIZE 784
#define MNIST_LABEL_SIZE 10
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

// Network architecture
#define HIDDEN_SIZE 128
#define LEARNING_RATE 0.01f
#define BATCH_SIZE 128
#define EPOCHS 10

// CUDA error checking macro
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// Structure for the neural network
typedef struct {
    float *h_input_weights;
    float *h_hidden_weights;
    float *h_hidden_bias;
    float *h_output_bias;
    
    // Device memory
    float *d_input_weights;
    float *d_hidden_weights;
    float *d_hidden_bias;
    float *d_output_bias;
    
    // Temporary storage for forward and backward pass
    float *d_hidden_output;
    float *d_output;
    float *d_output_error;
    float *d_hidden_error;
} NeuralNetwork;

// Function prototypes
void readMNISTData(float **train_images, unsigned char **train_labels, float **test_images, unsigned char **test_labels);
void initializeNetwork(NeuralNetwork *net);
void freeNetwork(NeuralNetwork *net);
void trainNetwork(NeuralNetwork *net, float *train_images, unsigned char *train_labels);
float testNetwork(NeuralNetwork *net, float *test_images, unsigned char *test_labels);

// CUDA kernels
__global__ void initializeWeights(float *weights, int size, unsigned long seed);
__global__ void forwardPass(float *images, float *input_weights, float *hidden_bias, float *hidden_output, float *hidden_weights, float *output_bias, float *output, int batch_size);
__global__ void calculateOutputError(float *output, unsigned char *labels, float *output_error, int batch_size);
__global__ void backpropagateError(float *output_error, float *hidden_weights, float *hidden_output, float *hidden_error, int batch_size);
__global__ void updateParameters(float *images, float *hidden_output, float *output_error, float *hidden_error, float *input_weights, float *hidden_weights, float *hidden_bias, float *output_bias, float learning_rate, int batch_size);
__global__ void predictDigits(float *output, int *predictions, int batch_size);

// ReLU activation function
__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

// Derivative of ReLU
__device__ float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Softmax function for output layer
__device__ void softmax(float *input, float *output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        max_val = fmaxf(max_val, input[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Read actual MNIST data from IDX files
void readMNISTData(float **train_images, unsigned char **train_labels, float **test_images, unsigned char **test_labels) {
    
    FILE *f_train_images = fopen("/kaggle/input/mydata2/train-images.idx3-ubyte", "rb");
    FILE *f_train_labels = fopen("/kaggle/input/mydata2/train-labels.idx1-ubyte", "rb");
    FILE *f_test_images = fopen("/kaggle/input/mydata2/t10k-images.idx3-ubyte", "rb");
    FILE *f_test_labels = fopen("/kaggle/input/mydata2/t10k-labels.idx1-ubyte", "rb");
    
    if(!f_train_images || !f_train_labels || !f_test_images || !f_test_labels) {
        printf("Error: Could not open one or more MNIST files.\n");
        exit(EXIT_FAILURE);
    }
    
    // Helper to reverse bytes (from big-endian to little-endian)
    void reverse_bytes(char *bytes, int size);
    
    int magic = 0, num = 0, rows = 0, cols = 0;
    
    // Read training images header
    fread(&magic, sizeof(magic), 1, f_train_images);
    reverse_bytes((char*)&magic, sizeof(magic));
    fread(&num, sizeof(num), 1, f_train_images);
    reverse_bytes((char*)&num, sizeof(num));
    fread(&rows, sizeof(rows), 1, f_train_images);
    reverse_bytes((char*)&rows, sizeof(rows));
    fread(&cols, sizeof(cols), 1, f_train_images);
    reverse_bytes((char*)&cols, sizeof(cols));
    
    if (rows * cols != MNIST_IMAGE_SIZE) {
        printf("Error: Unexpected image size.\n");
        exit(EXIT_FAILURE);
    }
    
    *train_images = (float*)malloc(num * MNIST_IMAGE_SIZE * sizeof(float));
    unsigned char *img_buffer = (unsigned char*)malloc(MNIST_IMAGE_SIZE * sizeof(unsigned char));
    
    for (int i = 0; i < num; i++) {
        fread(img_buffer, sizeof(unsigned char), MNIST_IMAGE_SIZE, f_train_images);
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            (*train_images)[i * MNIST_IMAGE_SIZE + j] = (float)img_buffer[j] / 255.0f;
        }
    }
    
    free(img_buffer);
    fclose(f_train_images);
    
    // Read training labels header
    fread(&magic, sizeof(magic), 1, f_train_labels);
    reverse_bytes((char*)&magic, sizeof(magic));
    fread(&num, sizeof(num), 1, f_train_labels);
    reverse_bytes((char*)&num, sizeof(num));
    
    *train_labels = (unsigned char*)malloc(num * sizeof(unsigned char));
    fread(*train_labels, sizeof(unsigned char), num, f_train_labels);
    fclose(f_train_labels);
    
    // Read test images header
    fread(&magic, sizeof(magic), 1, f_test_images);
    reverse_bytes((char*)&magic, sizeof(magic));
    fread(&num, sizeof(num), 1, f_test_images);
    reverse_bytes((char*)&num, sizeof(num));
    fread(&rows, sizeof(rows), 1, f_test_images);
    reverse_bytes((char*)&rows, sizeof(rows));
    fread(&cols, sizeof(cols), 1, f_test_images);
    reverse_bytes((char*)&cols, sizeof(cols));
    
    if (rows * cols != MNIST_IMAGE_SIZE) {
        printf("Error: Unexpected test image size.\n");
        exit(EXIT_FAILURE);
    }
    
    *test_images = (float*)malloc(num * MNIST_IMAGE_SIZE * sizeof(float));
    img_buffer = (unsigned char*)malloc(MNIST_IMAGE_SIZE * sizeof(unsigned char));
    
    for (int i = 0; i < num; i++) {
        fread(img_buffer, sizeof(unsigned char), MNIST_IMAGE_SIZE, f_test_images);
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            (*test_images)[i * MNIST_IMAGE_SIZE + j] = (float)img_buffer[j] / 255.0f;
        }
    }
    
    free(img_buffer);
    fclose(f_test_images);
    
    // Read test labels header
    fread(&magic, sizeof(magic), 1, f_test_labels);
    reverse_bytes((char*)&magic, sizeof(magic));
    fread(&num, sizeof(num), 1, f_test_labels);
    reverse_bytes((char*)&num, sizeof(num));
    
    *test_labels = (unsigned char*)malloc(num * sizeof(unsigned char));
    fread(*test_labels, sizeof(unsigned char), num, f_test_labels);
    fclose(f_test_labels);
    
    printf("MNIST data loaded: %d training images, %d test images.\n", TRAIN_SIZE, TEST_SIZE);
}

// Utility to reverse bytes from big-endian to little-endian
void reverse_bytes(char *bytes, int size) {
    for (int i = 0; i < size/2; i++) {
        char tmp = bytes[i];
        bytes[i] = bytes[size - i - 1];
        bytes[size - i - 1] = tmp;
    }
}

// Initialize the neural network with random weights (using Xavier initialization)
void initializeNetwork(NeuralNetwork *net) {

    // Allocate host memory for weights if needed
    net->h_input_weights = (float*)malloc(MNIST_IMAGE_SIZE * HIDDEN_SIZE * sizeof(float));
    net->h_hidden_weights = (float*)malloc(HIDDEN_SIZE * MNIST_LABEL_SIZE * sizeof(float));
    net->h_hidden_bias = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    net->h_output_bias = (float*)malloc(MNIST_LABEL_SIZE * sizeof(float));
    
    // Allocate device memory
    cudaMalloc((void**)&net->d_input_weights, MNIST_IMAGE_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&net->d_hidden_weights, HIDDEN_SIZE * MNIST_LABEL_SIZE * sizeof(float));
    cudaMalloc((void**)&net->d_hidden_bias, HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&net->d_output_bias, MNIST_LABEL_SIZE * sizeof(float));
    
    // Allocate temporary storage
    cudaMalloc((void**)&net->d_hidden_output, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&net->d_output, BATCH_SIZE * MNIST_LABEL_SIZE * sizeof(float));
    cudaMalloc((void**)&net->d_output_error, BATCH_SIZE * MNIST_LABEL_SIZE * sizeof(float));
    cudaMalloc((void**)&net->d_hidden_error, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaCheckError();
    
    // Initialize weights and biases on device with random values using a kernel
    dim3 blockSize(256);
    dim3 gridSize((MNIST_IMAGE_SIZE * HIDDEN_SIZE + blockSize.x - 1) / blockSize.x);
    initializeWeights<<<gridSize, blockSize>>> (net->d_input_weights, MNIST_IMAGE_SIZE * HIDDEN_SIZE, time(NULL));
    
    gridSize.x = (HIDDEN_SIZE * MNIST_LABEL_SIZE + blockSize.x - 1) / blockSize.x;
    initializeWeights<<<gridSize, blockSize>>> (net->d_hidden_weights, HIDDEN_SIZE * MNIST_LABEL_SIZE, time(NULL) + 1);
    
    gridSize.x = (HIDDEN_SIZE + blockSize.x - 1) / blockSize.x;
    initializeWeights<<<gridSize, blockSize>>> (net->d_hidden_bias, HIDDEN_SIZE, time(NULL) + 2);
    
    gridSize.x = (MNIST_LABEL_SIZE + blockSize.x - 1) / blockSize.x;
    initializeWeights<<<gridSize, blockSize>>> (net->d_output_bias, MNIST_LABEL_SIZE, time(NULL) + 3);
    cudaCheckError();
}

// CUDA kernel to initialize weights using curand and Xavier initialization
__global__ void initializeWeights(float *weights, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        float scale = 0.05f;
        weights[idx] = curand_normal(&state) * scale;
    }
}

// Free network memory
void freeNetwork(NeuralNetwork *net) {
    free(net->h_input_weights);
    free(net->h_hidden_weights);
    free(net->h_hidden_bias);
    free(net->h_output_bias);
    
    cudaFree(net->d_input_weights);
    cudaFree(net->d_hidden_weights);
    cudaFree(net->d_hidden_bias);
    cudaFree(net->d_output_bias);
    
    cudaFree(net->d_hidden_output);
    cudaFree(net->d_output);
    cudaFree(net->d_output_error);
    cudaFree(net->d_hidden_error);
}

// Training and testing functions (unchanged from your original code)
void trainNetwork(NeuralNetwork *net, float *train_images, unsigned char *train_labels) {
    float *d_train_images;
    unsigned char *d_train_labels;
    cudaMalloc((void**)&d_train_images, BATCH_SIZE * MNIST_IMAGE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_train_labels, BATCH_SIZE * sizeof(unsigned char));
    
    int *d_predictions;
    cudaMalloc((void**)&d_predictions, BATCH_SIZE * sizeof(int));
    
    dim3 forwardBlockSize(32, 4);
    dim3 forwardGridSize((HIDDEN_SIZE + forwardBlockSize.x - 1) / forwardBlockSize.x, (BATCH_SIZE + forwardBlockSize.y - 1) / forwardBlockSize.y);
    
    dim3 errorBlockSize(32, 4);
    dim3 errorGridSize((MNIST_LABEL_SIZE + errorBlockSize.x - 1) / errorBlockSize.x, (BATCH_SIZE + errorBlockSize.y - 1) / errorBlockSize.y);
    
    dim3 backpropBlockSize(32, 4);
    dim3 backpropGridSize((HIDDEN_SIZE + backpropBlockSize.x - 1) / backpropBlockSize.x, (BATCH_SIZE + backpropBlockSize.y - 1) / backpropBlockSize.y);
    
    dim3 updateBlockSize(32, 4);
    dim3 updateGridSize1((MNIST_IMAGE_SIZE + updateBlockSize.x - 1) / updateBlockSize.x, (HIDDEN_SIZE + updateBlockSize.y - 1) / updateBlockSize.y);
    dim3 updateGridSize2((HIDDEN_SIZE + updateBlockSize.x - 1) / updateBlockSize.x, (MNIST_LABEL_SIZE + updateBlockSize.y - 1) / updateBlockSize.y);
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("Epoch %d/%d\n", epoch + 1, EPOCHS);
        
        int *indices = (int*)malloc(TRAIN_SIZE * sizeof(int));
        for (int i = 0; i < TRAIN_SIZE; i++) indices[i] = i;
        
        for (int i = 0; i < TRAIN_SIZE - 1; i++) {
            int j = i + rand() % (TRAIN_SIZE - i);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        int num_batches = TRAIN_SIZE / BATCH_SIZE;
        int correct = 0;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Prepare batch data
            for (int i = 0; i < BATCH_SIZE; i++) {
                int idx = indices[batch * BATCH_SIZE + i];
                memcpy(&train_images[i * MNIST_IMAGE_SIZE], &train_images[idx * MNIST_IMAGE_SIZE], MNIST_IMAGE_SIZE * sizeof(float));
                train_labels[i] = train_labels[idx];
            }
            
            cudaMemcpy(d_train_images, train_images, BATCH_SIZE * MNIST_IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_train_labels, train_labels, BATCH_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
            
            forwardPass<<<forwardGridSize, forwardBlockSize>>>(
                d_train_images, net->d_input_weights, net->d_hidden_bias,
                net->d_hidden_output, net->d_hidden_weights, net->d_output_bias,
                net->d_output, BATCH_SIZE
            );
            cudaCheckError();
            
            calculateOutputError<<<errorGridSize, errorBlockSize>>>(
                net->d_output, d_train_labels, net->d_output_error, BATCH_SIZE
            );
            cudaCheckError();
            
            backpropagateError<<<backpropGridSize, backpropBlockSize>>>(
                net->d_output_error, net->d_hidden_weights, net->d_hidden_output,
                net->d_hidden_error, BATCH_SIZE
            );
            cudaCheckError();
            
            updateParameters<<<updateGridSize1, updateBlockSize>>>(
                d_train_images, net->d_hidden_output, net->d_output_error,
                net->d_hidden_error, net->d_input_weights, net->d_hidden_weights,
                net->d_hidden_bias, net->d_output_bias, LEARNING_RATE, BATCH_SIZE
            );
            cudaCheckError();
            
            predictDigits<<<(BATCH_SIZE + 255) / 256, 256>>>(
                net->d_output, d_predictions, BATCH_SIZE
            );
            cudaCheckError();
            
            int *h_predictions = (int*)malloc(BATCH_SIZE * sizeof(int));
            cudaMemcpy(h_predictions, d_predictions,
                      BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < BATCH_SIZE; i++) {
                if (h_predictions[i] == train_labels[i]) {
                    correct++;
                }
            }
            
            free(h_predictions);
            
            if ((batch + 1) % 50 == 0 || batch == num_batches - 1) {
                printf(" Batch %d/%d - Accuracy: %.2f%%\r",
                      batch + 1, num_batches,
                      100.0f * correct / ((batch + 1) * BATCH_SIZE));
                fflush(stdout);
            }
        }
        
        printf("\n");
        free(indices);
    }
    
    cudaMemcpy(net->h_input_weights, net->d_input_weights, MNIST_IMAGE_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->h_hidden_weights, net->d_hidden_weights, HIDDEN_SIZE * MNIST_LABEL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->h_hidden_bias, net->d_hidden_bias, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->h_output_bias, net->d_output_bias, MNIST_LABEL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_train_images);
    cudaFree(d_train_labels);
    cudaFree(d_predictions);
}

float testNetwork(NeuralNetwork *net, float *test_images, unsigned char *test_labels) {
    float *d_test_images;
    unsigned char *d_test_labels;
    cudaMalloc((void**)&d_test_images, BATCH_SIZE * MNIST_IMAGE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_test_labels, BATCH_SIZE * sizeof(unsigned char));
    
    int *d_predictions;
    cudaMalloc((void**)&d_predictions, BATCH_SIZE * sizeof(int));
    
    dim3 forwardBlockSize(32, 4);
    dim3 forwardGridSize((HIDDEN_SIZE + forwardBlockSize.x - 1) / forwardBlockSize.x, (BATCH_SIZE + forwardBlockSize.y - 1) / forwardBlockSize.y);
    
    int num_batches = TEST_SIZE / BATCH_SIZE;
    int correct = 0;
    
    for (int batch = 0; batch < num_batches; batch++) {
        cudaMemcpy(d_test_images, &test_images[batch * BATCH_SIZE * MNIST_IMAGE_SIZE], BATCH_SIZE * MNIST_IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_test_labels, &test_labels[batch * BATCH_SIZE], BATCH_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
        
        forwardPass<<<forwardGridSize, forwardBlockSize>>>(
            d_test_images, net->d_input_weights, net->d_hidden_bias,
            net->d_hidden_output, net->d_hidden_weights, net->d_output_bias,
            net->d_output, BATCH_SIZE
        );
        cudaCheckError();
        
        predictDigits<<<(BATCH_SIZE + 255) / 256, 256>>>(
            net->d_output, d_predictions, BATCH_SIZE
        );
        cudaCheckError();
        
        int *h_predictions = (int*)malloc(BATCH_SIZE * sizeof(int));
        cudaMemcpy(h_predictions, d_predictions, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < BATCH_SIZE; i++) {
            if (h_predictions[i] == test_labels[batch * BATCH_SIZE + i]) {
                correct++;
            }
        }
        
        free(h_predictions);
        
        if ((batch + 1) % 10 == 0 || batch == num_batches - 1) {
            printf(" Batch %d/%d - Accuracy: %.2f%%\r",
                  batch + 1, num_batches,
                  100.0f * correct / ((batch + 1) * BATCH_SIZE));
            fflush(stdout);
        }
    }
    
    printf("\n");
    cudaFree(d_test_images);
    cudaFree(d_test_labels);
    cudaFree(d_predictions);
    
    return (float)correct / (num_batches * BATCH_SIZE);
}

// CUDA kernel implementations
__global__ void forwardPass(float *images, float *input_weights, float *hidden_bias, float *hidden_output, float *hidden_weights, float *output_bias, float *output, int batch_size) {
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (hidden_idx < HIDDEN_SIZE && batch_idx < batch_size) {
        float sum = hidden_bias[hidden_idx];
        for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
            sum += images[batch_idx * MNIST_IMAGE_SIZE + i] * input_weights[i * HIDDEN_SIZE + hidden_idx];
        }
        hidden_output[batch_idx * HIDDEN_SIZE + hidden_idx] = relu(sum);
    }
    
    __syncthreads();
    
    if (hidden_idx < MNIST_LABEL_SIZE && batch_idx < batch_size) {
        float sum = output_bias[hidden_idx];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum += hidden_output[batch_idx * HIDDEN_SIZE + i] * hidden_weights[i * MNIST_LABEL_SIZE + hidden_idx];
        }
        output[batch_idx * MNIST_LABEL_SIZE + hidden_idx] = sum;
    }
    
    __syncthreads();
    
    if (hidden_idx == 0 && batch_idx < batch_size) {
        softmax(&output[batch_idx * MNIST_LABEL_SIZE], &output[batch_idx * MNIST_LABEL_SIZE], MNIST_LABEL_SIZE);
    }
}

__global__ void calculateOutputError(float *output, unsigned char *labels, float *output_error, int batch_size) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (output_idx < MNIST_LABEL_SIZE && batch_idx < batch_size) {
        float target = (labels[batch_idx] == output_idx) ? 1.0f : 0.0f;
        output_error[batch_idx * MNIST_LABEL_SIZE + output_idx] = output[batch_idx * MNIST_LABEL_SIZE + output_idx] - target;
    }
}

__global__ void backpropagateError(float *output_error, float *hidden_weights, float *hidden_output, float *hidden_error, int batch_size) {
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (hidden_idx < HIDDEN_SIZE && batch_idx < batch_size) {
        float error = 0.0f;
        for (int i = 0; i < MNIST_LABEL_SIZE; i++) {
            error += output_error[batch_idx * MNIST_LABEL_SIZE + i] * hidden_weights[hidden_idx * MNIST_LABEL_SIZE + i];
        }
        hidden_error[batch_idx * HIDDEN_SIZE + hidden_idx] = error * relu_derivative(hidden_output[batch_idx * HIDDEN_SIZE + hidden_idx]);
    }
}

__global__ void updateParameters(float *images, float *hidden_output, float *output_error, float *hidden_error, float *input_weights, float *hidden_weights, float *hidden_bias, float *output_bias, float learning_rate, int batch_size) {
    
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hidden_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (input_idx < MNIST_IMAGE_SIZE && hidden_idx < HIDDEN_SIZE) {
        float weight_gradient = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            weight_gradient += images[b * MNIST_IMAGE_SIZE + input_idx] * hidden_error[b * HIDDEN_SIZE + hidden_idx];
        }
        input_weights[input_idx * HIDDEN_SIZE + hidden_idx] -= learning_rate * weight_gradient / batch_size;
    }
    
    if (input_idx == 0 && hidden_idx < HIDDEN_SIZE) {
        float bias_gradient = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            bias_gradient += hidden_error[b * HIDDEN_SIZE + hidden_idx];
        }
        hidden_bias[hidden_idx] -= learning_rate * bias_gradient / batch_size;
    }
    
    if (hidden_idx < HIDDEN_SIZE && input_idx < MNIST_LABEL_SIZE) {
        float weight_gradient = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            weight_gradient += hidden_output[b * HIDDEN_SIZE + hidden_idx] * output_error[b * MNIST_LABEL_SIZE + input_idx];
        }
        hidden_weights[hidden_idx * MNIST_LABEL_SIZE + input_idx] -= learning_rate * weight_gradient / batch_size;
    }
    
    if (hidden_idx == 0 && input_idx < MNIST_LABEL_SIZE) {
        float bias_gradient = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            bias_gradient += output_error[b * MNIST_LABEL_SIZE + input_idx];
        }
        output_bias[input_idx] -= learning_rate * bias_gradient / batch_size;
    }
}

__global__ void predictDigits(float *output, int *predictions, int batch_size) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float max_prob = output[batch_idx * MNIST_LABEL_SIZE];
        int max_idx = 0;
        
        for (int i = 1; i < MNIST_LABEL_SIZE; i++) {
            float prob = output[batch_idx * MNIST_LABEL_SIZE + i];
            if (prob > max_prob) {
                max_prob = prob;
                max_idx = i;
            }
        }
        
        predictions[batch_idx] = max_idx;
    }
}

int main() {
    
    // Load MNIST data from files
    float *train_images, *test_images;
    unsigned char *train_labels, *test_labels;
    
    printf("Loading MNIST dataset...\n");
    readMNISTData(&train_images, &train_labels, &test_images, &test_labels);
    
    // Initialize neural network
    NeuralNetwork net;
    printf("Initializing neural network...\n");
    initializeNetwork(&net);
    
    // Train network
    printf("Training neural network...\n");
    clock_t train_start = clock();
    trainNetwork(&net, train_images, train_labels);
    clock_t train_end = clock();
    printf("Training time: %lf seconds\n", (double)(train_end - train_start) / CLOCKS_PER_SEC);
    
    printf("Testing neural network...\n");
    clock_t test_start = clock();
    float acc = testNetwork(&net, test_images, test_labels);
    clock_t test_end = clock();
    printf("Testing time: %lf seconds\n", (double)(test_end - test_start) / CLOCKS_PER_SEC);
    
    printf("Final accuracy: %.2f%%\n", acc * 100.0f);
    
    // Clean up
    freeNetwork(&net);
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);
    
    return 0;
}