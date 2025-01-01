#include <iostream>
#include "NN.h"

using namespace std;

#define IMGSIZE 28*28  // change depending of the size of the images used

int main () {
    string train_data_csv_path = "mnist/training_paths.csv";  // a cpp file is provided in utilities to generate this files
    string test_data_csv_path = "mnist/testing_paths.csv";

    // training parameters
    int epochs = 10;
    int batch_size = 32;
    float learning_rate = 0.001;

    string model_path = "saved_model/model.bin";

    // choose one of the following functions to run
    // trainModel (model_path, train_data_csv_path, 60000, batch_size, IMGSIZE, epochs, learning_rate);
    // improveModel (model_path, train_data_csv_path, 60000, batch_size, IMGSIZE, epochs, learning_rate);
    // testModel (model_path, test_data_csv_path, 1000, IMGSIZE);
    // predict(model_path, "test_img.png", IMGSIZE);

    return 0;
}