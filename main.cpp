#include<iostream>
#include<fstream>
#include<vector>
#include<iomanip>
#include<cmath>
#include "src/Neuron.hpp"
#include "src/MSE.hpp"
#include "src/Layer.hpp"
#include "src/Sigmoid.hpp"


using namespace std;

void reverse_bytes(char *bytes, int size){

    for(int i = 0; i < size / 2; i++){
        char temp = bytes[i];
        bytes[i] = bytes[size - i - 1];
        bytes[size - i - 1] = temp;
    }
}


bool load_data(vector<vector<double>> *train_images, vector<int> *train_labels, vector<vector<double>> *test_images, vector<int> *test_labels){

    //train-labels-idex1-ubyte
    ifstream file_labels;
    file_labels.open("dataset/train-labels.idx1-ubyte", ios:: binary | ios:: in);

    if(!file_labels.is_open()){
        return false;
    }

    //seek start of the file.
    file_labels.seekg(0, ios::beg);

    //First 32 bits are magic number and 32 bits are no of items.
    
    int magic_number;
    file_labels.read((char *)&magic_number, sizeof(magic_number));
    reverse_bytes((char *)&magic_number, sizeof(magic_number));

    int no_of_items;
    file_labels.read((char *)&no_of_items, sizeof(no_of_items));
    reverse_bytes((char *)&no_of_items, sizeof(no_of_items));

    //Read the labels.

    for(int i = 0; i < no_of_items; i++){

        //Each label is 1 byte so we use char
        char label;
        file_labels.read(&label, sizeof(label));
        train_labels->push_back((int)label);
    }

    file_labels.close();

    ifstream images;
    images.open("dataset/train-images.idx3-ubyte", ios::binary | ios::in);

    if(!images.is_open()){
        return false;
    }

    //seek to beginning
    images.seekg(0, ios::beg);

    //First 32 - magic number
    //Next 32 - no of images
    //Next 32 - no of rows
    //Next 32 - no of columns

    //read magic number
    int magic_number_images;
    images.read((char *)&magic_number_images, sizeof(magic_number_images));
    reverse_bytes((char *)&magic_number_images, sizeof(magic_number_images));

    //read no of images
    int no_of_images;
    images.read((char *)&no_of_images, sizeof(no_of_images));
    reverse_bytes((char *)&no_of_images, sizeof(no_of_images));

    //read no of rows
    int no_of_rows;
    images.read((char *)&no_of_rows, sizeof(no_of_rows));
    reverse_bytes((char *)&no_of_rows, sizeof(no_of_rows));

    //read no of columns
    int no_of_columns;
    images.read((char *)&no_of_columns, sizeof(no_of_columns));
    reverse_bytes((char *)&no_of_columns, sizeof(no_of_columns));

    //read train_images
    for(int i = 0; i < no_of_images; i++){

        //each image is 28 * 28 = 784 bytes
        char image[784];
        images.read(image, 784);

        vector<double> image_vector(784);

        for(int j = 0; j < 784; j++){
            unsigned int temp = (unsigned int)(unsigned char)image[j];

            //Normalize the pixel values to 0-1
            image_vector[j] = (double)temp / 255.0;
        }

        train_images->push_back(image_vector);
    }

    images.close();

    //Load test labels
    ifstream test_labels_file;
    test_labels_file.open("dataset/t10k-labels.idx1-ubyte", ios::binary | ios::in);
    if(!test_labels_file.is_open()){
        return false;
    }

    test_labels_file.seekg(0, ios::beg);

    int test_magic_number;
    test_labels_file.read((char *)&test_magic_number, sizeof(test_magic_number));
    reverse_bytes((char *)&test_magic_number, sizeof(test_magic_number));

    int test_no_of_items;
    test_labels_file.read((char *)&test_no_of_items, sizeof(test_no_of_items));
    reverse_bytes((char *)&test_no_of_items, sizeof(test_no_of_items));

    for(int i = 0; i < test_no_of_items; i++){

        char label;
        test_labels_file.read(&label, sizeof(label));
        test_labels->push_back((int)label);
    }

    test_labels_file.close();

    //Load test_images

    ifstream test_images_file;
    test_images_file.open("dataset/t10k-images.idx3-ubyte", ios::binary | ios::in);
    if(!test_images_file.is_open()){
        return false;
    }

    test_images_file.seekg(0, ios::beg);

    int test_magic_number_images;
    test_images_file.read((char *)&test_magic_number_images, sizeof(test_magic_number_images));
    reverse_bytes((char *)&test_magic_number_images, sizeof(test_magic_number_images));

    int test_no_of_images;
    test_images_file.read((char *)&test_no_of_images, sizeof(test_no_of_images));
    reverse_bytes((char *)&test_no_of_images, sizeof(test_no_of_images));

    int test_no_of_rows;
    test_images_file.read((char *)&test_no_of_rows, sizeof(test_no_of_rows));
    reverse_bytes((char *)&test_no_of_rows, sizeof(test_no_of_rows));

    int test_no_of_columns;
    test_images_file.read((char *)&test_no_of_columns, sizeof(test_no_of_columns));
    reverse_bytes((char *)&test_no_of_columns, sizeof(test_no_of_columns));

    for(int i = 0; i < test_no_of_images; i++){
        char image[784];
        test_images_file.read(image, 784);

        vector<double> image_vector(784);

        for(int j = 0; j < 784; j++){
            unsigned int temp = (unsigned int)(unsigned char)image[j];
            image_vector[j] = (double)temp / 255.0;
        }

        test_images->push_back(image_vector);
    }

    test_images_file.close();

    return true;
}

double accuracy(vector<int> &predictions, vector<int> &labels){

    int correct = 0;
    for(size_t i = 0; i < predictions.size(); i++){
        if(predictions[i] == labels[i]){
            correct++;
        }
    }

    return (double)correct / (double)predictions.size();
}

int main(void){

    srand(time(0));

    vector<vector<double>> train_images;
    vector<int> train_labels;

    vector<vector<double>> test_images;
    vector<int> test_labels;

    bool flag = load_data(&train_images, &train_labels, &test_images, &test_labels);

    if(!flag){
        cout << "Error loading data" << endl;
        return 1;
    }
    
    cout << "Data loaded successfully" << endl;
    cout << "Training images: " << train_images.size() << endl;
    cout << "Training labels: " << train_labels.size() << endl;
    cout << "Test images: " << test_images.size() << endl;
    cout << "Test labels: " << test_labels.size() << endl;

    Layer l1 = Layer(784, 100);
    Sigmoid s1 = Sigmoid();
    Layer l2 = Layer(100, 10);
    Sigmoid s2 = Sigmoid();


    for(int epoch = 0; epoch < 5; epoch++){
        double learning_rate = 0.1;
        double mean_loss = 0.0;

        size_t i = 0;
        vector<int> predictions;

        for(; i < train_images.size(); i++){
            int idx = i;
            vector<double> image = train_images[idx];
            int label = train_labels[idx];

            vector<double> l1_output = l1.feed_forward(image);
            vector<double> s1_output = s1.feed_forward(l1_output);
            vector<double> l2_output = l2.feed_forward(s1_output);
            vector<double> s2_output = s2.feed_forward(l2_output);

            vector<double> target(10, 0.0);
            target[label] = 1.0;

            int prediction = 0;
            for(size_t j = 0; j < s2_output.size(); j++){
                if(s2_output[j] > s2_output[prediction]){
                    prediction = j;
                }
            }

            predictions.push_back(prediction);

            MSE loss = MSE();
            double loss_value = loss.feed_forward(s2_output, target);

            mean_loss += loss_value;
            if(i % 500 == 0){
                cout << setprecision(4) << "Epoch: " << epoch + 1 << " | Mean loss: " << mean_loss / (i + 1) << "\r" << flush;
            }

            //Backpropagation

            //Zero grad for every layer
            l1.zero_grad();
            l2.zero_grad();

            loss.backward(1.0);

            //Backward pass
            s2.backward(loss.grad);
            l2.backward(s2.grad);
            s1.backward(l2);
            l1.backward(s1.grad);

            //Update weights
            l1.descend(learning_rate);
            l2.descend(learning_rate);

        }

        double acc = accuracy(predictions, train_labels);

        cout << "                                      \r" 
            << "Epoch: " << epoch + 1 << " | Loss: " << mean_loss / i << " | Training accuracy: " << acc * 100 << endl;
    }


    //Test the model

    vector<int> predictions;

    for(size_t i = 0; i < test_images.size(); i++){
        int idx = i;
        vector<double> image = test_images[idx];
        //int label = test_labels[idx];

        //Feed forward
        vector<double> l1_output = l1.feed_forward(image);
        vector<double> s1_output = s1.feed_forward(l1_output);
        vector<double> l2_output = l2.feed_forward(s1_output);
        vector<double> s2_output = s2.feed_forward(l2_output);

        //Prediction is the index of the maximum value in the output
        int prediction = 0;
        
        for(size_t j = 0; j < s2_output.size(); j++){
            if(s2_output[j] > s2_output[prediction]){
                prediction = j;
            }
        }

        predictions.push_back(prediction);
    }

    double acc = accuracy(predictions, test_labels);
    cout << "Test accuracy: " << acc * 100  << endl;

    return 0;
}

