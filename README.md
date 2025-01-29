# Parallelization of Neural Networks using OpenMP

### Overview
This repository contains code to implement a **Neural Network** from scratch in **C**, specifically designed for the **MNIST** dataset, with an emphasis on **parallelization** using **OpenMP** for improving performance on multi-core processors.

### Prerequisites
Before you begin, ensure you have the following installed:

- **C Compiler** (e.g., GCC)
- **OpenMP** (Make sure your compiler supports OpenMP)
- **MNIST Dataset** (Either download it from [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) or use a local copy)
- **Make** (Optional, if you want to use a makefile for compilation)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ashrithiiitdm/Parallel_NNCpp.git
   cd Parallel_NNCpp.git
   ```

2. Compile and run the code:
    ```bash
    make
    ```
3. Or compile using the required flags:
    ```bash
    gcc -o main main.c -lm
    ./main
    ```