#include <iostream>
#include "NN.h"
#include "knn.h"

using namespace std;

#define IMGSIZE 28*28  // change depending of the size of the images used

int main () {
    // creating the neural network
    vector<Layer> NN;
    addLayer(NN, "ReLU", 256, IMGSIZE);
    addLayer(NN, "ReLU", 128);
    addLayer(NN, "softmax", 10);
    
    // example use of other activation functions
    // addLayer(NN, "ELU", 128, vector<float>{1.0f});
    // addLayer(NN, "SELU", 64);
    // addLayer(NN, "SELU", 32, vector<float>{1.0f, 1.0f});
    // addLayer(NN, "sigmoid", 10);

    string train_data_csv_path = "mnist/training_paths.csv";  // a cpp file is provided in utilities to generate this files
    string test_data_csv_path = "mnist/testing_paths.csv";

    // training parameters
    int epochs = 30;
    int batch_size = 32;
    float learning_rate = 0.001;  // start with 0.01 then improve with 0.001

    string model_path = "saved_model/model.bin";

    // choose one of the following functions to run
    // trainModel (NN, model_path, train_data_csv_path, 60000, batch_size, epochs, learning_rate);
    // improveModel (NN, model_path, train_data_csv_path, 60000, batch_size, epochs, learning_rate);
    // testModel (NN, model_path, test_data_csv_path, 10000);
    // predict(NN, model_path, "test_img.png");

    if (NN.size() > 0) {
        destroyNN(NN);
    }

   /*
    // KNN example
    string img_path = "test_img.png";
    string data_csv_path = "mnist/training_paths.csv";
    int k = 5;
    int data_size = 60000;
    int class_nb = 10;
    knn(img_path, k, data_csv_path, data_size, class_nb);
    */

    return 0;
}