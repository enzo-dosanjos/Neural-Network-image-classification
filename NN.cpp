/*************************************************************************
NN - implement a basic neural network with one hidden layer for now
                             -------------------
    copyright            : (C) 2024 by Enzo DOS ANJOS
*************************************************************************/

//-------- Implementation of the module <utilsNN> (file utilsNN.cpp) ---------

/////////////////////////////////////////////////////////////////  INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
using namespace std;

//------------------------------------------------------- Personal Include
#include "NN.h"
#include "utilsNN.h"
#include "readFile.h"

/////////////////////////////////////////////////////////////////  PRIVATE
//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//------------------------------------------------------- Static Variables

//------------------------------------------------------ Private Functions

//////////////////////////////////////////////////////////////////  PUBLIC
//------------------------------------------------------- Public Functions

void initLayer (float *layer_weight, float *layer_bias, int input_size, int layer_size) {
    // Initialize the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // initialise weight array with random values
    for (int i = 0; i < input_size * layer_size; i++) {
        layer_weight[i] = static_cast<float>(rand()) / RAND_MAX * 0.2f - 0.1f; // Uniform distribution between -0.1 and 0.1;

        if (i < layer_size) {
            layer_bias[i] = 0; // Initialise biases to 0
        }
    }
} //----- end of initLayer

void computeLayer (float *input, float *layer_weight, float *layer_bias, int input_size, int layer_size, float *output, string activation_function) {
    mulMat(input, input_size, layer_weight, layer_size, output);
    addMat(output, layer_bias, layer_size, output);

    if (activation_function == "ReLU") {
        ReLU(output, layer_size);
    }
    else if (activation_function == "softmax") {
        softmax(output, output, layer_size);
    }  // todo: add other activation functions
    else {
        cout << "Activation function not recognised" << endl;
    }
} //----- end of computeLayer

void computeError (float *output, int label, float *error) {
    error[label] = output[label] - 1;  // 1 is the wanted output for the label
    for (int i = 0; i < 10; i++) {
        if (i != label) {
            error[i] = output[i];  // 0 is the wanted output for the other classes
        }
    }
} //----- end of computeError

void trainModel (string model_path, string data_csv_path, int data_size, int batch_size, int input_size, int epochs, float learning_rate) {
    float *imgArray = nullptr;
    int width, height, channels;

    float epoch_loss;
    float batch_loss;
    float loss;

    // Initialise weight and bias arrays
    // layer 1
    int layer_1_size = 256;
    float layer_1_weight[input_size * layer_1_size];
    float layer_1_bias[layer_1_size];
    float layer_1_res[layer_1_size];

    initLayer(layer_1_weight, layer_1_bias, input_size, layer_1_size);

    // output
    int output_size = 10;
    float output_weight[layer_1_size * output_size];
    float output_bias[output_size];
    float output[output_size];

    initLayer(output_weight, output_bias, layer_1_size, output_size);

    // Read the training data
    Data dataArray[data_size];
    readData(data_csv_path, dataArray, data_size);

    for (int e = 0; e < epochs; e++) {
        shuffleData(dataArray, data_size);

        epoch_loss = 0;

        for (int d = 0; d < data_size; d+=batch_size) {
            float layer_1_weight_gradient[input_size * layer_1_size] = {0};
            float layer_1_bias_gradient[layer_1_size] = {0};
            float output_weight_gradient[layer_1_size * output_size] = {0};
            float output_bias_gradient[output_size] = {0};

            batch_loss = 0;

            for (int b = d; b < (data_size < d + batch_size ? data_size : d + batch_size) ; b++) {
                if (readImg(dataArray[b].path, imgArray, width, height, channels)) {
                    // Normalise the image
                    Normalize(imgArray, width, height, 28, 28, channels);

                    int label = dataArray[b].label;

                    // layer 1
                    computeLayer(imgArray, layer_1_weight, layer_1_bias, input_size, layer_1_size, layer_1_res, "ReLU");

                    // output
                    computeLayer(layer_1_res, output_weight, output_bias, layer_1_size, output_size, output, "softmax");

                    // loss
                    loss = crossEntropyLoss(output, label);
                    epoch_loss += loss;
                    batch_loss += loss;

                    // backpropagation
                    // Initialise output error with the error for each class
                    float output_error[output_size];
                    computeError(output, label, output_error);

                    // Compute the error for the first layer
                    float layer_1_error[layer_1_size];
                    backPropagation(output_weight, output_size, output_error, layer_1_res, layer_1_size, layer_1_error, "ReLU");

                    // Accumulate Gradients
                    accumulateGradient(output_weight_gradient, output_bias_gradient, output_error, layer_1_res, layer_1_size, output_size);
                    accumulateGradient(layer_1_weight_gradient, layer_1_bias_gradient, layer_1_error, imgArray, input_size, layer_1_size);

                    // Free the image memory when done
                    delete[] imgArray;
                
                } else {
                    cerr << "Error: Failed to read image" << endl;
                }
            }
            updateWeights(output_weight, output_bias, output_weight_gradient, output_bias_gradient, layer_1_size, output_size, learning_rate);
            updateWeights(layer_1_weight, layer_1_bias, layer_1_weight_gradient, layer_1_bias_gradient, input_size, layer_1_size, learning_rate);                    
                
            cout << "Batch's average loss: " << batch_loss/batch_size << " | " << ((float)d/(float)data_size)*100.0 << "%" << endl;
        }

        cout << "Epoch " << e+1 << "/" << epochs << " done. Average loss: " << epoch_loss/data_size << endl;
    }

    // Save the model
    saveModel(model_path, input_size, layer_1_weight, layer_1_bias, layer_1_size, output_weight, output_bias, output_size);
} //----- end of trainModel


void improveModel (string model_path, string data_csv_path, int data_size, int batch_size, int input_size, int epochs, float learning_rate) {
    float *imgArray;
    int width, height, channels;

    float epoch_loss;
    float batch_loss;
    float loss;

    // Initialise weight and bias arrays
    // layer 1
    int layer_1_size = 256;
    float layer_1_weight[input_size * layer_1_size];
    float layer_1_bias[layer_1_size];
    float layer_1_res[layer_1_size];

    // output
    int output_size = 10;
    float output_weight[layer_1_size * output_size];
    float output_bias[output_size];
    float output[output_size];

    // load model from file
    loadModel(model_path, input_size, layer_1_weight, layer_1_bias, layer_1_size, output_weight, output_bias, output_size);

    // Read the training data
    Data dataArray[data_size];
    readData(data_csv_path, dataArray, data_size);

    for (int e = 0; e < epochs; e++) {
        shuffleData(dataArray, data_size);

        epoch_loss = 0;

        for (int d = 0; d < data_size; d+=batch_size) {
            float layer_1_weight_gradient[input_size * layer_1_size] = {0};
            float layer_1_bias_gradient[layer_1_size] = {0};
            float output_weight_gradient[layer_1_size * output_size] = {0};
            float output_bias_gradient[output_size] = {0};

            batch_loss = 0;

            for (int b = d; b < (data_size < d + batch_size ? data_size : d + batch_size) ; b++) {
                if (readImg(dataArray[b].path, imgArray, width, height, channels)) {
                    // Normalise the image
                    Normalize(imgArray, width, height, 28, 28, channels);

                    int label = dataArray[b].label;

                    // layer 1
                    computeLayer(imgArray, layer_1_weight, layer_1_bias, input_size, layer_1_size, layer_1_res, "ReLU");

                    // output
                    computeLayer(layer_1_res, output_weight, output_bias, layer_1_size, output_size, output, "softmax");

                    // loss
                    loss = crossEntropyLoss(output, label);
                    epoch_loss += loss;
                    batch_loss += loss;

                    // backpropagation
                    // Initialise output error with the error for each class
                    float output_error[output_size];
                    computeError(output, label, output_error);

                    // Compute the error for the first layer
                    float layer_1_error[layer_1_size];
                    backPropagation(output_weight, output_size, output_error, layer_1_res, layer_1_size, layer_1_error, "ReLU");

                    // Accumulate Gradients
                    accumulateGradient(output_weight_gradient, output_bias_gradient, output_error, layer_1_res, layer_1_size, output_size);
                    accumulateGradient(layer_1_weight_gradient, layer_1_bias_gradient, layer_1_error, imgArray, input_size, layer_1_size);

                    // Free the image memory when done
                    delete[] imgArray;

                } else {
                    cerr << "Error: Failed to read image" << endl;
                }
            }
            updateWeights(output_weight, output_bias, output_weight_gradient, output_bias_gradient, layer_1_size, output_size, learning_rate);
            updateWeights(layer_1_weight, layer_1_bias, layer_1_weight_gradient, layer_1_bias_gradient, input_size, layer_1_size, learning_rate);                    
                
            cout << "Batch's average loss: " << batch_loss/batch_size << " | " << ((float)d/(float)data_size)*100.0 << "%" << endl;
        }

        cout << "Epoch " << e+1 << "/" << epochs << " done. Average loss: " << epoch_loss/data_size << endl;
    }

    // Save the model
    saveModel(model_path, input_size, layer_1_weight, layer_1_bias, layer_1_size, output_weight, output_bias, output_size);
} //----- end of improveModel


void testModel (string model_path, string data_csv_path, int data_size, int input_size) {
    int correct_predictions = 0;

    float *imgArray;
    int width, height, channels;

    // Read weight and bias arrays from saved model
    // layer 1
    int layer_1_size = 256;
    float layer_1_weight[input_size * layer_1_size];
    float layer_1_bias[layer_1_size];
    float layer_1_res[layer_1_size];

    // output
    int output_size = 10;
    float output_weight[layer_1_size * output_size];
    float output_bias[output_size];
    float output[output_size];

    // load model from file
    loadModel(model_path, input_size, layer_1_weight, layer_1_bias, layer_1_size, output_weight, output_bias, output_size);

    // Read the training data
    Data dataArray[data_size];
    readData(data_csv_path, dataArray, data_size);

    for (int d = 0; d < data_size; d++) {
        if (readImg(dataArray[d].path, imgArray, width, height, channels)) {
            // Normalise the image
            Normalize(imgArray, width, height, 28, 28, channels);

            int label = dataArray[d].label;

            // layer 1
            computeLayer(imgArray, layer_1_weight, layer_1_bias, input_size, layer_1_size, layer_1_res, "ReLU");

            // output
            computeLayer(layer_1_res, output_weight, output_bias, layer_1_size, output_size, output, "softmax");

            float highest_prob = 0;
            int predicted_label = 0;
            for (int i = 0; i < output_size; i++) {
                if (output[i] > highest_prob) {
                    highest_prob = output[i];
                    predicted_label = i;
                }
            }

            if (predicted_label == label) {
                correct_predictions++;
                cout << "Correct prediction. Total: " << correct_predictions << endl;
            } else {
                cout << "Incorrect prediction. Total: " << correct_predictions << endl;
            }

            // Free the image memory when done
            delete[] imgArray;
        
        } else {
            cerr << "Error: Failed to read image" << endl;
        }
    }

    cout << "Accuracy: " << ((float)correct_predictions/(float)data_size)*100.0 << "%" << endl;
} //----- end of testModel


void predict (string model_path, string img_path, int input_size) {
    float *imgArray;
    int width, height, channels;

    // Read weight and bias arrays from saved model
    // layer 1
    int layer_1_size = 256;
    float layer_1_weight[input_size * layer_1_size];
    float layer_1_bias[layer_1_size];
    float layer_1_res[layer_1_size];

    // output
    int output_size = 10;
    float output_weight[layer_1_size * output_size];
    float output_bias[output_size];
    float output[output_size];

    // load model from file
    loadModel(model_path, input_size, layer_1_weight, layer_1_bias, layer_1_size, output_weight, output_bias, output_size);

    if (readImg(img_path, imgArray, width, height, channels)) {
        // Normalise the image
        Normalize(imgArray, width, height, 28, 28, channels);

        // layer 1
        computeLayer(imgArray, layer_1_weight, layer_1_bias, input_size, layer_1_size, layer_1_res, "ReLU");

        // output
        computeLayer(layer_1_res, output_weight, output_bias, layer_1_size, output_size, output, "softmax");

        float highest_prob = 0;
        int predicted_label = 0;
        for (int i = 0; i < output_size; i++) {
            if (output[i] > highest_prob) {
                highest_prob = output[i];
                predicted_label = i;
            }
        }

        // Free the image memory when done
        delete[] imgArray;

        cout << "Predicted label: " << predicted_label << "  with " << highest_prob*100.0 << "%" << endl;

    } else {
        cerr << "Error: Failed to read image" << endl;
    }
} //----- end of predict