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

// store the NN layers
#include <vector>

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


void initLayerGradient (float *layer_weight_grad, float *layer_bias_grad, int input_size, int layer_size) {
    for (int i = 0; i < input_size * layer_size; i++) {
        layer_weight_grad[i] = 0; // Initialise gradients to 0

        if (i < layer_size) {
            layer_bias_grad[i] = 0;
        }
    }
} //----- end of initLayerGradient


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


void addLayer (vector<Layer> &NN, string type, int output_size, int input_size) {
    Layer layer;
    layer.type = type;

    layer.input_size = (NN.size() > 0) ? NN[NN.size()-1].output_size : input_size;

    layer.output_size = output_size;
    layer.weights = new float[layer.input_size * output_size];
    layer.biases = new float[output_size];
    layer.output = new float[output_size];

    layer.weight_gradient = new float[layer.input_size * output_size];
    layer.bias_gradient = new float[output_size];
    layer.error = new float[output_size];

    initLayer(layer.weights, layer.biases, layer.input_size, output_size);

    initLayerGradient(layer.weight_gradient, layer.bias_gradient, layer.input_size, output_size);

    NN.push_back(layer);
} //----- end of addLayer


void computeError (float *output, int label, float *error) {
    error[label] = output[label] - 1;  // 1 is the wanted output for the label
    for (int i = 0; i < 10; i++) {
        if (i != label) {
            error[i] = output[i];  // 0 is the wanted output for the other classes
        }
    }
} //----- end of computeError


void destroyNN(vector<Layer> &NN) {
    for (int i = 0; i < NN.size(); i++) {
        // Free allocated memory
        delete[] NN[i].weights;
        delete[] NN[i].biases;
        delete[] NN[i].output;

        delete[] NN[i].weight_gradient;
        delete[] NN[i].bias_gradient;
        delete[] NN[i].error;
    }
    NN.clear(); // Clear the vector to remove all elements
} //----- end of destroyNN


void trainModel (vector<Layer> &NN, string model_path, string data_csv_path, int data_size, int batch_size, int epochs, float learning_rate) {
    if (NN.size() < 1) {
        cerr << "Error: The neural network must have at least 1 layer" << endl;
        return;
    }

    float *imgArray;
    int width, height, channels;

    float epoch_loss;
    float batch_loss;
    float loss;

    // Read the training data
    Data dataArray[data_size];
    readData(data_csv_path, dataArray, data_size);

    for (int e = 0; e < epochs; e++) {
        shuffleData(dataArray, data_size);

        epoch_loss = 0;

        for (int d = 0; d < data_size; d+=batch_size) {
            batch_loss = 0;

            for (auto &layer : NN) {
                initLayerGradient(layer.weight_gradient, layer.bias_gradient, layer.input_size, layer.output_size);
            }

            for (int b = d; b < (data_size < d + batch_size ? data_size : d + batch_size) ; b++) {
                if (readImg(dataArray[b].path, imgArray, width, height, channels)) {
                    // Normalise the image
                    Normalize(imgArray, width, height, 28, 28, channels);

                    int label = dataArray[b].label;

                    for (int i = 0; i < NN.size(); i++) {
                        float *input = (i == 0) ? imgArray : NN[i - 1].output;
                        computeLayer(input, NN[i].weights, NN[i].biases, NN[i].input_size, NN[i].output_size, NN[i].output, NN[i].type);
                    }

                    // Compute loss for the last layer
                    Layer &output = NN.back();
                    float loss = crossEntropyLoss(output.output, label);
                    epoch_loss += loss;
                    batch_loss += loss;

                    // Backward pass
                    computeError(output.output, label, output.error);
                    for (int i = NN.size() - 1; i > 0; i--) {
                        backPropagation(NN[i].weights, NN[i].output_size, NN[i].error, NN[i - 1].output, NN[i - 1].output_size, NN[i - 1].error, NN[i - 1].type);
                    }

                    // Accumulate Gradients
                    for (int i = 0; i < NN.size(); i++) {
                        float *input = (i == 0) ? imgArray : NN[i - 1].output;
                        accumulateGradient(NN[i].weight_gradient, NN[i].bias_gradient, NN[i].error, input, NN[i].input_size, NN[i].output_size);
                    }

                    // Free the image memory when done
                    delete[] imgArray;
                
                } else {
                    cerr << "Error: Failed to read image" << endl;
                }
            }
            // Update weights and biases for all layers
            for (auto &layer : NN) {
                updateWeights(layer.weights, layer.biases, layer.weight_gradient, layer.bias_gradient, layer.input_size, layer.output_size, learning_rate);
            }      
                
            cout << "Batch's average loss: " << batch_loss/batch_size << " | " << ((float)d/(float)data_size)*100.0 << "%" << endl;
        }

        cout << "Epoch " << e+1 << "/" << epochs << " done. Average loss: " << epoch_loss/data_size << endl;
    }

    // Save the model
    saveModel(model_path, NN);

    destroyNN(NN);
} //----- end of trainModel


void improveModel (vector<Layer> &NN, string model_path, string data_csv_path, int data_size, int batch_size, int epochs, float learning_rate) {
    // load model from file
    loadModel(model_path, NN);

    trainModel(NN, model_path, data_csv_path, data_size, batch_size, epochs, learning_rate);
} //----- end of improveModel


int predictLabel (vector<Layer> &NN, string img_path, float &highest_prob) {
    float *imgArray;
    int width, height, channels;

    if (readImg(img_path, imgArray, width, height, channels)) {
        // Normalise the image
        Normalize(imgArray, width, height, 28, 28, channels);

        for (int i = 0; i < NN.size(); i++) {
            float *input = (i == 0) ? imgArray : NN[i - 1].output;
            computeLayer(input, NN[i].weights, NN[i].biases, NN[i].input_size, NN[i].output_size, NN[i].output, NN[i].type);
        }

        highest_prob = 0;
        int predicted_label = 0;
        for (int i = 0; i < NN.back().output_size; i++) {
            if (NN.back().output[i] > highest_prob) {
                highest_prob = NN.back().output[i];
                predicted_label = i;
            }
        }

        // Free the image memory when done
        delete[] imgArray;

        return predicted_label;

    } else {
        cerr << "Error: Failed to read image" << endl;
        return -1;
    }
} //----- end of predict


void testModel (vector<Layer> &NN, string model_path, string data_csv_path, int data_size) {
    int correct_predictions = 0;

    float *imgArray;
    int width, height, channels;

    // load model from file
    loadModel(model_path, NN);

    // Read the testing data
    Data dataArray[data_size];
    readData(data_csv_path, dataArray, data_size);

    for (int d = 0; d < data_size; d++) {
        int label = dataArray[d].label;

        float _prob;
        int predicted_label = predictLabel(NN, dataArray[d].path, _prob);

        if (predicted_label == label) {
            correct_predictions++;
            cout << "Correct prediction. Total: " << correct_predictions << endl;
        } else {
            cout << "Incorrect prediction. Total: " << correct_predictions << endl;
        }
    }

    cout << "Accuracy: " << ((float)correct_predictions/(float)data_size)*100.0 << "%" << endl;

    destroyNN(NN);
} //----- end of testModel


void predict (vector<Layer> &NN, string model_path, string img_path) {
    loadModel(model_path, NN);

    float highest_prob;
    int predicted_label = predictLabel(NN, img_path, highest_prob);

    cout << "Predicted label: " << predicted_label << "  with " << highest_prob*100.0 << "%" << endl;

    destroyNN(NN);
} //----- end of predict