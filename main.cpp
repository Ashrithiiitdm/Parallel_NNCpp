#include<iostream>
#include<fstream>
#include<vector>
#include<iomanip>
#include<cmath>
#include "src/Neuron.cpp"
#include "src/Layer.cpp"
#include "src/MSE.cpp"
#include "src/Sigmoid.cpp"
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

    

}

