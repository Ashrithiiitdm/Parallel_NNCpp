#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>

double get_random()
{
    return (rand() / (double)RAND_MAX) * 2 - 1;
}

typedef struct
{
    double *weights;
    double *wgrad;
    double bias;
    double bgrad;
    int input_size;
} Neuron;

void init_neuron(Neuron *n, int input_size)
{
    n->input_size = input_size;
    n->weights = (double *)calloc(input_size, sizeof(double));
    n->wgrad = (double *)calloc(input_size, sizeof(double));
    n->bias = 0.01 * get_random();
    n->bgrad = 0.0;

#pragma omp parallel for
    for (int i = 0; i < input_size; i++)
    {
        n->weights[i] = get_random();
    }
}

void free_neuron(Neuron *n)
{
    free(n->weights);
    free(n->wgrad);
}

void zero_grad(Neuron *n)
{
    for (int i = 0; i < n->input_size; i++)
    {
        n->wgrad[i] = 0.0;
    }
    n->bgrad = 0.0;
}

double feed_forward(Neuron *n, double *inputs)
{
    double sum = n->bias;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n->input_size; i++)
    {
        sum += inputs[i] * n->weights[i];
    }
    return sum;
}

void backpropagation(Neuron *n, double *last_input, double grad)
{
    n->bgrad += grad;
#pragma omp parallel for
    for (int i = 0; i < n->input_size; i++)
    {
        n->wgrad[i] += grad * last_input[i];
    }
}

void descend(Neuron *n, double learning_rate)
{
    n->bias -= learning_rate * n->bgrad;
#pragma omp parallel for
    for (int i = 0; i < n->input_size; i++)
    {
        n->weights[i] -= learning_rate * n->wgrad[i];
    }
}

typedef struct
{
    Neuron *neurons;
    double *last_input;
    int input_size;
    int output_size;
} Layer;

void init_layer(Layer *l, int input_size, int output_size)
{
    l->input_size = input_size;
    l->output_size = output_size;
    l->neurons = (Neuron *)malloc(output_size * sizeof(Neuron));
    l->last_input = (double *)malloc(input_size * sizeof(double));

#pragma omp parallel for
    for (int i = 0; i < output_size; i++)
    {
        init_neuron(&l->neurons[i], input_size);
    }
}

void free_layer(Layer *l)
{
    free(l->neurons);
    free(l->last_input);
}

void zero_grad_layer(Layer *l)
{
    for (int i = 0; i < l->output_size; i++)
    {
        zero_grad(&l->neurons[i]);
    }
}

double *feed_forward_layer(Layer *l, double *inputs)
{
    l->last_input = inputs;
    double *outputs = (double *)malloc(l->output_size * sizeof(double));

#pragma omp parallel for
    for (int i = 0; i < l->output_size; i++)
    {
        outputs[i] = feed_forward(&l->neurons[i], inputs);
    }

    return outputs;
}

void backward(Layer *l, double *grad)
{
#pragma omp parallel for
    for (int i = 0; i < l->output_size; i++)
    {
        backpropagation(&l->neurons[i], l->last_input, grad[i]);
    }
}

void descend_layer(Layer *l, double learning_rate)
{
#pragma omp parallel for
    for (int i = 0; i < l->output_size; i++)
    {
        descend(&l->neurons[i], learning_rate);
    }
}

typedef struct
{
    double *last_input;
    double *last_target;
    double *grad;
    int size;
} MSE;

void init_mse(MSE *mse)
{
    mse->last_input = NULL;
    mse->last_target = NULL;
    mse->grad = NULL;
    mse->size = 0;
}

void free_mse(MSE *mse)
{
    free(mse->last_input);
    free(mse->last_target);
    free(mse->grad);
}

double feed_forward_mse(MSE *mse, double *inputs, double *targets, int size)
{
    mse->last_input = (double *)malloc(size * sizeof(double));
    mse->last_target = (double *)malloc(size * sizeof(double));
    mse->size = size;

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        mse->last_input[i] = inputs[i];
        mse->last_target[i] = targets[i];
    }

    double sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < size; i++)
    {
        double s = mse->last_input[i] - mse->last_target[i];
        sum += s * s;
    }

    return sum / size;
}

void backward_mse(MSE *mse, double grad)
{
    mse->grad = (double *)malloc(mse->size * sizeof(double));
#pragma omp parallel for
    for (int i = 0; i < mse->size; i++)
    {
        mse->grad[i] = 2 * (mse->last_input[i] - mse->last_target[i]) / mse->size;
        mse->grad[i] *= grad;
    }
}

typedef struct
{
    double *last_input;
    double *grad;
    double *last_output;
    int size;
} Sigmoid;

void init_sigmoid(Sigmoid *s)
{
    s->last_input = NULL;
    s->grad = NULL;
    s->last_output = NULL;
    s->size = 0;
}

void free_sigmoid(Sigmoid *s)
{
    free(s->last_input);
    free(s->grad);
    free(s->last_output);
}

double *feed_forward_sigmoid(Sigmoid *s, double *inputs, int size)
{
    s->last_input = (double *)malloc(size * sizeof(double));
    s->last_output = (double *)malloc(size * sizeof(double));
    s->size = size;

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        s->last_input[i] = inputs[i];
    }

    double *outputs = s->last_output;
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        outputs[i] = 1 / (1 + exp(-inputs[i]));
    }

    return outputs;
}

void backward_chain(Sigmoid *s, double *chain_grad, int size)
{
    s->grad = (double *)malloc(size * sizeof(double));
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        s->grad[i] = s->last_output[i] * (1 - s->last_output[i]) * chain_grad[i];
    }
}

void backward_layer(Sigmoid *s, Layer *prev_layer)
{
    s->grad = (double *)malloc(s->size * sizeof(double));
#pragma omp parallel for
    for (int i = 0; i < s->size; i++)
    {
        double sum = 0;
        for (int j = 0; j < prev_layer->output_size; j++)
        {
            sum += prev_layer->neurons[j].weights[i] * prev_layer->neurons[j].wgrad[i];
        }
        s->grad[i] = s->last_output[i] * (1 - s->last_output[i]) * sum;
    }
}

void reverse_bytes(char *bytes, int size)
{
    for (int i = 0; i < size / 2; i++)
    {
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
#pragma omp parallel for reduction(+ : correct)
    for (int i = 0; i < size; i++)
    {
        if (predictions[i] == labels[i])
        {
            correct++;
        }
    }
    return (double)correct / (double)size;
}

int main(void)
{
    srand(time(0));

    double **train_images;
    int *train_labels;
    int train_size;

    double **test_images;
    int *test_labels;
    int test_size;

    bool flag = load_data(&train_images, &train_labels, &test_images, &test_labels, &train_size, &test_size);

    if (!flag)
    {
        printf("Error loading data\n");
        return 1;
    }

    printf("Data loaded successfully\n");
    printf("Training images: %d\n", train_size);
    printf("Training labels: %d\n", train_size);
    printf("Test images: %d\n", test_size);
    printf("Test labels: %d\n", test_size);

    Layer l1, l2;
    init_layer(&l1, 784, 100);
    init_layer(&l2, 100, 10);
    Sigmoid s1, s2;
    init_sigmoid(&s1);
    init_sigmoid(&s2);

    double start_time = omp_get_wtime();

    for (int epoch = 0; epoch < 10; epoch++)
    {
        double learning_rate = 0.1;
        double mean_loss = 0.0;

        int *predictions = (int *)malloc(train_size * sizeof(int));

        for (int i = 0; i < train_size; i++)
        {
            int idx = i;
            double *image = train_images[idx];
            int label = train_labels[idx];

            double *l1_output = feed_forward_layer(&l1, image);
            double *s1_output = feed_forward_sigmoid(&s1, l1_output, 100);
            double *l2_output = feed_forward_layer(&l2, s1_output);
            double *s2_output = feed_forward_sigmoid(&s2, l2_output, 10);

            double target[10] = {0.0};
            target[label] = 1.0;

            int prediction = 0;
            for (int j = 0; j < 10; j++)
            {
                if (s2_output[j] > s2_output[prediction])
                {
                    prediction = j;
                }
            }

            predictions[i] = prediction;

            MSE loss;
            double loss_value = feed_forward_mse(&loss, s2_output, target, 10);

            mean_loss += loss_value;
            if (i % 500 == 0)
            {
                printf("Epoch: %d | Mean loss: %.4f\r", epoch + 1, mean_loss / (i + 1));
            }

            zero_grad_layer(&l1);
            zero_grad_layer(&l2);

            backward_mse(&loss, 1.0);

            backward_chain(&s2, loss.grad, 10);
            backward(&l2, s2.grad);
            backward_layer(&s1, &l2);
            backward(&l1, s1.grad);

            descend_layer(&l1, learning_rate);
            descend_layer(&l2, learning_rate);
        }

        double acc = accuracy(predictions, train_labels, train_size);

        printf("Epoch: %d | Loss: %.4f | Training accuracy: %.2f%%\n", epoch + 1, mean_loss / train_size, acc * 100);

        free(predictions);
    }
    double end_time = omp_get_wtime();

    printf("Time for training: %f", end_time - start_time);

    int *test_predictions = (int *)malloc(test_size * sizeof(int));

    start_time = omp_get_wtime();

    for (int i = 0; i < test_size; i++)
    {
        double *image = test_images[i];

        double *l1_output = feed_forward_layer(&l1, image);
        double *s1_output = feed_forward_sigmoid(&s1, l1_output, 100);
        double *l2_output = feed_forward_layer(&l2, s1_output);
        double *s2_output = feed_forward_sigmoid(&s2, l2_output, 10);

        int prediction = 0;
        for (int j = 0; j < 10; j++)
        {
            if (s2_output[j] > s2_output[prediction])
            {
                prediction = j;
            }
        }

        test_predictions[i] = prediction;
    }

    end_time = omp_get_wtime();

    double test_acc = accuracy(test_predictions, test_labels, test_size);
    printf("Test accuracy: %.2f%%\n", test_acc * 100);
    printf("Time for testing: %f", end_time - start_time);
    free(test_predictions);
    return 0;
}
