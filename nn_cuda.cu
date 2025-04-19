#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<stdbool.h>
#include<time.h>
#include<cuda.h>
#include <cuda_runtime.h>

// 1. Fix random number generation for device
__device__ double get_random_device(unsigned long long seed) {
    // Simple LCG PRNG for device
    seed = (seed * 6364136223846793005ULL + 1442695040888963407ULL);
    return (double)(seed >> 32) / (double)0xFFFFFFFFULL * 2.0 - 1.0;
}

__host__ __device__ double get_random(unsigned long long seed) {
#ifdef __CUDA_ARCH__
    // Device version
    return get_random_device(seed);
#else
    // Host version
    return (rand() / (double)RAND_MAX) * 2 - 1;
#endif
}

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

typedef struct{
    double *weights;
    double *wgrad;
    double bias;
    double bgrad;
    int input_size;
} Neuron;

__global__ void init_neuron_kernel(Neuron *n, int input_size, unsigned long long seed){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < input_size){
        n->weights[i] = get_random_device(seed + 1);
    }
}

void init_neuron(Neuron *n, int input_size){

    n->input_size = input_size;
    cudaMallocManaged(&n->weights, input_size * sizeof(double));
    cudaMallocManaged(&n->wgrad, input_size * sizeof(double));
    n->bias = 0.01 * get_random(time(NULL));
    n->bgrad = 0.0;
    unsigned long long seed = time(NULL);
    int blockSize = 256;
    int numBlocks = (input_size + blockSize - 1) / blockSize;
    init_neuron_kernel<<<numBlocks, blockSize>>>(n, input_size, seed);
    cudaDeviceSynchronize();
}

void free_neuron(Neuron *n){

    cudaFree(n->weights);
    cudaFree(n->wgrad);
}

__global__ void zero_grad_kernel(Neuron *n){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n->input_size){
        n->wgrad[i] = 0.0;
    }

    if (i == 0){
        n->bgrad = 0.0;
    }
}

void zero_grad(Neuron *n){

    int blockSize = 256;
    int numBlocks = (n->input_size + blockSize - 1) / blockSize;
    zero_grad_kernel<<<numBlocks, blockSize>>>(n);
    cudaDeviceSynchronize();
}

__global__ void feed_forward_kernel(Neuron *n, double *inputs, double *sum){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n->input_size){
        atomicAddDouble(sum, inputs[i] * n->weights[i]);
    }
}

__device__ double feed_forward(Neuron *n, double *inputs){

    double sum = n->bias;
    double *d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemcpy(d_sum, &sum, sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n->input_size + blockSize - 1) / blockSize;
    feed_forward_kernel<<<numBlocks, blockSize>>>(n, inputs, d_sum);
    cudaDeviceSynchronize();

    cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    return sum;
}

__global__ void backpropagation_kernel(Neuron *n, double *last_input, double grad){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n->input_size){
        atomicAddDouble(&n->wgrad[i], grad * last_input[i]);
    }
    if (i == 0){
        atomicAddDouble(&n->bgrad, grad);
    }
}

void backpropagation(Neuron *n, double *last_input, double grad){

    int blockSize = 256;
    int numBlocks = (n->input_size + blockSize - 1) / blockSize;
    backpropagation_kernel<<<numBlocks, blockSize>>>(n, last_input, grad);
    cudaDeviceSynchronize();
}

__global__ void descend_kernel(Neuron *n, double learning_rate){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n->input_size){
        n->weights[i] -= learning_rate * n->wgrad[i];
    }
    if (i == 0){
        n->bias -= learning_rate * n->bgrad;
    }
}

void descend(Neuron *n, double learning_rate){

    int blockSize = 256;
    int numBlocks = (n->input_size + blockSize - 1) / blockSize;
    descend_kernel<<<numBlocks, blockSize>>>(n, learning_rate);
    cudaDeviceSynchronize();
}

typedef struct{

    Neuron *neurons;
    double *last_input;
    int input_size;
    int output_size;
} Layer;

void init_layer(Layer *l, int input_size, int output_size){

    l->input_size = input_size;
    l->output_size = output_size;
    cudaMallocManaged(&l->neurons, output_size * sizeof(Neuron));
    cudaMallocManaged(&l->last_input, input_size * sizeof(double));

    for (int i = 0; i < output_size; i++){
        init_neuron(&l->neurons[i], input_size);
    }
}

void free_layer(Layer *l){

    for (int i = 0; i < l->output_size; i++){
        free_neuron(&l->neurons[i]);
    }
    cudaFree(l->neurons);
    cudaFree(l->last_input);
}

void zero_grad_layer(Layer *l){

    for (int i = 0; i < l->output_size; i++){
        zero_grad(&l->neurons[i]);
    }
}

__global__ void feed_forward_layer_kernel(Layer *l, double *inputs, double *outputs){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < l->output_size){
        outputs[i] = feed_forward(&l->neurons[i], inputs);
    }
}

double *feed_forward_layer(Layer *l, double *inputs){

    l->last_input = inputs;
    double *outputs;
    cudaMallocManaged(&outputs, l->output_size * sizeof(double));

    int blockSize = 256;
    int numBlocks = (l->output_size + blockSize - 1) / blockSize;
    feed_forward_layer_kernel<<<numBlocks, blockSize>>>(l, inputs, outputs);
    cudaDeviceSynchronize();

    return outputs;
}

void backward(Layer *l, double *grad){

    for (int i = 0; i < l->output_size; i++){
        backpropagation(&l->neurons[i], l->last_input, grad[i]);
    }
}

void descend_layer(Layer *l, double learning_rate){

    for (int i = 0; i < l->output_size; i++){
        descend(&l->neurons[i], learning_rate);
    }
}

typedef struct{

    double *last_input;
    double *last_target;
    double *grad;
    int size;
} MSE;

void init_mse(MSE *mse){

    mse->last_input = NULL;
    mse->last_target = NULL;
    mse->grad = NULL;
    mse->size = 0;
}

void free_mse(MSE *mse){

    cudaFree(mse->last_input);
    cudaFree(mse->last_target);
    cudaFree(mse->grad);
}

__global__ void feed_forward_mse_kernel(MSE *mse, double *d_sum){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < mse->size){
        double s = mse->last_input[i] - mse->last_target[i];
        atomicAddDouble(d_sum, s * s);
    }
}

double feed_forward_mse(MSE *mse, double *inputs, double *targets, int size){

    cudaMallocManaged(&mse->last_input, size * sizeof(double));
    cudaMallocManaged(&mse->last_target, size * sizeof(double));
    mse->size = size;

    cudaMemcpy(mse->last_input, inputs, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mse->last_target, targets, size * sizeof(double), cudaMemcpyHostToDevice);

    double sum = 0;
    double *d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemcpy(d_sum, &sum, sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    feed_forward_mse_kernel<<<numBlocks, blockSize>>>(mse, d_sum);
    cudaDeviceSynchronize();

    cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);

    return sum / size;
}



__global__ void backward_mse_kernel(MSE *mse, double grad){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < mse->size){
        mse->grad[i] = 2 * (mse->last_input[i] - mse->last_target[i]) / mse->size;
        mse->grad[i] *= grad;
    }
}

void backward_mse(MSE *mse, double grad){

    cudaMallocManaged(&mse->grad, mse->size * sizeof(double));

    int blockSize = 256;
    int numBlocks = (mse->size + blockSize - 1) / blockSize;
    backward_mse_kernel<<<numBlocks, blockSize>>>(mse, grad);
    cudaDeviceSynchronize();
}

typedef struct{
    double *last_input;
    double *grad;
    double *last_output;
    int size;
} Sigmoid;

void init_sigmoid(Sigmoid *s){

    s->last_input = NULL;
    s->grad = NULL;
    s->last_output = NULL;
    s->size = 0;
}

void free_sigmoid(Sigmoid *s){

    cudaFree(s->last_input);
    cudaFree(s->grad);
    cudaFree(s->last_output);
}

__global__ void feed_forward_sigmoid_kernel(Sigmoid *s, double *outputs){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < s->size){
        outputs[i] = 1 / (1 + exp(-s->last_input[i]));
    }
}

double *feed_forward_sigmoid(Sigmoid *s, double *inputs, int size){

    cudaMallocManaged(&s->last_input, size * sizeof(double));
    cudaMallocManaged(&s->last_output, size * sizeof(double));
    s->size = size;

    cudaMemcpy(s->last_input, inputs, size * sizeof(double), cudaMemcpyHostToDevice);

    double *outputs = s->last_output;

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    feed_forward_sigmoid_kernel<<<numBlocks, blockSize>>>(s, outputs);
    cudaDeviceSynchronize();

    return outputs;
}



__global__ void backward_chain_kernel(Sigmoid *s, double *chain_grad){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < s->size){
        s->grad[i] = s->last_output[i] * (1 - s->last_output[i]) * chain_grad[i];
    }
}

void backward_chain(Sigmoid *s, double *chain_grad, int size){

    cudaMallocManaged(&s->grad, size * sizeof(double));

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    backward_chain_kernel<<<numBlocks, blockSize>>>(s, chain_grad);
    cudaDeviceSynchronize();
}

__global__ void backward_layer_kernel(Sigmoid *s, Layer *prev_layer){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < s->size){

        double sum = 0;
        for (int j = 0; j < prev_layer->output_size; j++){
            sum += prev_layer->neurons[j].weights[i] * prev_layer->neurons[j].wgrad[i];
        }
        s->grad[i] = s->last_output[i] * (1 - s->last_output[i]) * sum;
    }
}

void backward_layer(Sigmoid *s, Layer *prev_layer){

    cudaMallocManaged(&s->grad, s->size * sizeof(double));

    int blockSize = 256;
    int numBlocks = (s->size + blockSize - 1) / blockSize;
    backward_layer_kernel<<<numBlocks, blockSize>>>(s, prev_layer);
    cudaDeviceSynchronize();
}

void reverse_bytes(char *bytes, int size){

    for (int i = 0; i < size / 2; i++){
        char temp = bytes[i];
        bytes[i] = bytes[size - i - 1];
        bytes[size - i - 1] = temp;
    }
}

bool load_data(double ***train_images, int **train_labels, double ***test_images, int **test_labels, int *train_size, int *test_size)
{
    FILE *file_labels = fopen("dataset/train-labels.idx1-ubyte", "rb");
    if (!file_labels)
    {
        return false;
    }

    int magic_number, no_of_items;
    fread(&magic_number, sizeof(magic_number), 1, file_labels);
    reverse_bytes((char *)&magic_number, sizeof(magic_number));
    fread(&no_of_items, sizeof(no_of_items), 1, file_labels);
    reverse_bytes((char *)&no_of_items, sizeof(no_of_items));

    *train_labels = (int *)malloc(no_of_items * sizeof(int));
    for (int i = 0; i < no_of_items; i++)
    {
        char label;
        fread(&label, sizeof(label), 1, file_labels);
        (*train_labels)[i] = (int)label;
    }
    fclose(file_labels);

    FILE *images = fopen("dataset/train-images.idx3-ubyte", "rb");
    if (!images)
    {
        return false;
    }

    int magic_number_images, no_of_images, no_of_rows, no_of_columns;
    fread(&magic_number_images, sizeof(magic_number_images), 1, images);
    reverse_bytes((char *)&magic_number_images, sizeof(magic_number_images));
    fread(&no_of_images, sizeof(no_of_images), 1, images);
    reverse_bytes((char *)&no_of_images, sizeof(no_of_images));
    fread(&no_of_rows, sizeof(no_of_rows), 1, images);
    reverse_bytes((char *)&no_of_rows, sizeof(no_of_rows));
    fread(&no_of_columns, sizeof(no_of_columns), 1, images);
    reverse_bytes((char *)&no_of_columns, sizeof(no_of_columns));

    *train_images = (double **)malloc(no_of_images * sizeof(double *));
    for (int i = 0; i < no_of_images; i++)
    {
        char image[784];
        fread(image, sizeof(image), 1, images);
        (*train_images)[i] = (double *)malloc(784 * sizeof(double));
        for (int j = 0; j < 784; j++)
        {
            (*train_images)[i][j] = (double)((unsigned char)image[j]) / 255.0;
        }
    }
    fclose(images);
    *train_size = no_of_images;

    FILE *test_labels_file = fopen("dataset/t10k-labels.idx1-ubyte", "rb");
    if (!test_labels_file)
    {
        return false;
    }

    int test_magic_number, test_no_of_items;
    fread(&test_magic_number, sizeof(test_magic_number), 1, test_labels_file);
    reverse_bytes((char *)&test_magic_number, sizeof(test_magic_number));
    fread(&test_no_of_items, sizeof(test_no_of_items), 1, test_labels_file);
    reverse_bytes((char *)&test_no_of_items, sizeof(test_no_of_items));

    *test_labels = (int *)malloc(test_no_of_items * sizeof(int));
    for (int i = 0; i < test_no_of_items; i++)
    {
        char label;
        fread(&label, sizeof(label), 1, test_labels_file);
        (*test_labels)[i] = (int)label;
    }
    fclose(test_labels_file);

    FILE *test_images_file = fopen("dataset/t10k-images.idx3-ubyte", "rb");
    if (!test_images_file)
    {
        return false;
    }

    int test_magic_number_images, test_no_of_images, test_no_of_rows, test_no_of_columns;
    fread(&test_magic_number_images, sizeof(test_magic_number_images), 1, test_images_file);
    reverse_bytes((char *)&test_magic_number_images, sizeof(test_magic_number_images));
    fread(&test_no_of_images, sizeof(test_no_of_images), 1, test_images_file);
    reverse_bytes((char *)&test_no_of_images, sizeof(test_no_of_images));
    fread(&test_no_of_rows, sizeof(test_no_of_rows), 1, test_images_file);
    reverse_bytes((char *)&test_no_of_rows, sizeof(test_no_of_rows));
    fread(&test_no_of_columns, sizeof(test_no_of_columns), 1, test_images_file);
    reverse_bytes((char *)&test_no_of_columns, sizeof(test_no_of_columns));

    *test_images = (double **)malloc(test_no_of_images * sizeof(double *));
    for (int i = 0; i < test_no_of_images; i++)
    {
        char image[784];
        fread(image, sizeof(image), 1, test_images_file);
        (*test_images)[i] = (double *)malloc(784 * sizeof(double));
        for (int j = 0; j < 784; j++)
        {
            (*test_images)[i][j] = (double)((unsigned char)image[j]) / 255.0;
        }
    }
    fclose(test_images_file);
    *test_size = test_no_of_images;

    return true;
}

double accuracy(int *predictions, int *labels, int size)
{
    int correct = 0;
    for (int i = 0; i < size; i++)
    {
        if (predictions[i] == labels[i])
        {
            correct++;
        }
    }
    return (double)correct / (double)size;
}


// Main function with training loop (complete implementation)
int main() {
    srand(time(0));
    
    // Load MNIST data
    double **train_images, **test_images;
    int *train_labels, *test_labels;
    int train_size, test_size;
    
    if(!load_data(&train_images, &train_labels, &test_images, &test_labels, &train_size, &test_size)) {
        printf("Failed to load data!\n");
        return 1;
    }
    printf("Loaded %d training samples and %d test samples\n", train_size, test_size);

    // Initialize network
    Layer l1, l2;
    init_layer(&l1, 784, 128);
    init_layer(&l2, 128, 10);
    Sigmoid s1, s2;
    init_sigmoid(&s1);
    init_sigmoid(&s2);
    MSE loss;
    init_mse(&loss);

    // Training parameters
    const int epochs = 10;
    const double learning_rate = 0.1;
    const int batch_size = 100;

    // Training loop
    clock_t start = clock();
    for(int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        int *predictions = (int*)malloc(train_size * sizeof(int));

        for(int i = 0; i < train_size; i++) {
            // Forward pass
            double *l1_out = feed_forward_layer(&l1, train_images[i]);
            double *s1_out = feed_forward_sigmoid(&s1, l1_out, 128);
            double *l2_out = feed_forward_layer(&l2, s1_out);
            double *s2_out = feed_forward_sigmoid(&s2, l2_out, 10);

            // Get prediction
            int pred = 0;
            for(int j = 0; j < 10; j++) {
                if(s2_out[j] > s2_out[pred]) pred = j;
            }
            predictions[i] = pred;

            // Calculate loss
            double target[10] = {0};
            target[train_labels[i]] = 1.0;
            double loss_val = feed_forward_mse(&loss, s2_out, target, 10);
            epoch_loss += loss_val;

            // Backpropagation
            zero_grad_layer(&l1);
            zero_grad_layer(&l2);
            
            backward_mse(&loss, 1.0);
            backward_chain(&s2, loss.grad, 10);
            backward(&l2, s2.grad);
            backward_layer(&s1, &l2);
            backward(&l1, s1.grad);

            // Update weights
            descend_layer(&l1, learning_rate);
            descend_layer(&l2, learning_rate);

            if(i % 1000 == 0) {
                printf("Epoch %d: Sample %d/%d - Loss: %.4f\r", 
                      epoch+1, i+1, train_size, epoch_loss/(i+1));
            }
        }

        // Calculate epoch metrics
        double acc = accuracy(predictions, train_labels, train_size);
        printf("Epoch %d complete - Loss: %.4f - Accuracy: %.2f%%\n",
              epoch+1, epoch_loss/train_size, acc*100);
        free(predictions);
    }
    double train_time = (double)(clock() - start)/CLOCKS_PER_SEC;

    // Testing
    start = clock();
    int *test_preds = (int*)malloc(test_size * sizeof(int));
    for(int i = 0; i < test_size; i++) {
        double *l1_out = feed_forward_layer(&l1, test_images[i]);
        double *s1_out = feed_forward_sigmoid(&s1, l1_out, 128);
        double *l2_out = feed_forward_layer(&l2, s1_out);
        double *s2_out = feed_forward_sigmoid(&s2, l2_out, 10);

        int pred = 0;
        for(int j = 0; j < 10; j++) {
            if(s2_out[j] > s2_out[pred]) pred = j;
        }
        test_preds[i] = pred;
    }
    double test_acc = accuracy(test_preds, test_labels, test_size);
    double test_time = (double)(clock() - start)/CLOCKS_PER_SEC;

    printf("\nFinal Results:\n");
    printf("Training Time: %.2f seconds\n", train_time);
    printf("Test Accuracy: %.2f%%\n", test_acc * 100);
    printf("Inference Time: %.2f seconds\n", test_time);

    // Cleanup
    free_layer(&l1);
    free_layer(&l2);
    free_sigmoid(&s1);
    free_sigmoid(&s2);
    free_mse(&loss);
    
    // Free data arrays
    for(int i = 0; i < train_size; i++) free(train_images[i]);
    for(int i = 0; i < test_size; i++) free(test_images[i]);
    free(train_images);
    free(test_images);
    free(train_labels);
    free(test_labels);
    free(test_preds);

    return 0;
}
