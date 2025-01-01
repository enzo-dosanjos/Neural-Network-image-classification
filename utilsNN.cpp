/*************************************************************************
utilsNN - utility functions for neural network operations
                             -------------------
    copyright            : (C) 2024 by Enzo DOS ANJOS
*************************************************************************/

//-------- Implementation of the module <utilsNN> (file utilsNN.cpp) ---------

/////////////////////////////////////////////////////////////////  INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
using namespace std;

//------------------------------------------------------- Personal Include

// softmax
#include "utilsNN.h"
#include <cmath>

// shuffle
#include <algorithm>
#include <random>

/////////////////////////////////////////////////////////////////  PRIVATE
//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//------------------------------------------------------- Static Variables

//------------------------------------------------------ Private Functions

//////////////////////////////////////////////////////////////////  PUBLIC
//------------------------------------------------------- Public Functions
void ReLU(float *vals, int len) {
    for (int i = 0; i < len; i++) {
        if (vals[i] < 0) {
            vals[i] = 0;
        }
    }
} //----- end of ReLU


void softmax(float *vals, float *new_vals, int len) {
    float Z = 0;
    float temp[len];
    for (int i = 0; i < len; i++) {
        temp[i] = exp(vals[i]);
        Z += temp[i];
    }

    for (int i = 0; i < len; i++) {
        new_vals[i] = temp[i]/Z;
    }
} //----- end of softmax


void mulMat(float *input, int input_size, float *neuron_weight, int nb_neuron, float *res) {
    for (int j = 0; j < nb_neuron; j++) { // For each neuron (column in neuron_weight)
        float sum = 0;
        for (int i = 0; i < input_size; i++) { // For each input
            sum += input[i] * neuron_weight[i * nb_neuron + j];
        }
        res[j] = sum;
    }
} //----- end of mulMat


void addMat(float *input, float *bias, int len, float *res) {
    for (int i = 0; i < len; i++) {
        res[i] = input[i] + bias[i];
    }
} //----- end of addMat


float crossEntropyLoss (float *output, const int label) {
    float epsilon = 1e-10; // small value to avoid log(0)
    float loss = -log(output[label] + epsilon);
    return loss;
} //----- end of crossEntropyLoss


void backPropagation(float *next_layer_weight, int next_layer_size, float *next_layer_error, float *layer_output, int layer_size, float *layer_error, string activation_function) {
    for (int i = 0; i < layer_size; i++) {
        layer_error[i] = 0; // Initialize the error to 0 for each neuron in the current layer
        for (int j = 0; j < next_layer_size; j++) {
            layer_error[i] += next_layer_weight[i * next_layer_size + j] * next_layer_error[j];
        }

        if (activation_function == "ReLU") {
            layer_error[i] *= (layer_output[i] > 0 ? 1 : 0); // Derivative of ReLU

        } /*else if (activation_function == "softmax") {  todo: other useful activation functions
        }*/ else {
            cerr << "Error: unrecognized activation function" << endl;
        }
    }
} //----- end of backPropagation


void accumulateGradient(float *weights_gradient, float *biases_gradient, float *error, float *layer_input, int input_size, int layer_size) {
    // Accumulate the gradients
    for (int i = 0; i < layer_size; i++) { // For each output neuron
        for (int j = 0; j < input_size; j++) { // For each input neuron
            weights_gradient[j * layer_size + i] += error[i] * layer_input[j];
            
        }

        // Accumulate the bias gradient
        biases_gradient[i] += error[i];
    }
} //----- end of accumulateGradient


void updateWeights(float *weights, float *biases, float *weights_gradient, float *biases_gradient, int input_size, int layer_size, float learning_rate) {
    for (int i = 0; i < layer_size; i++) { // For each neuron of the layer
        for (int j = 0; j < input_size; j++) { // For each input
            weights[j * layer_size + i] -= learning_rate * weights_gradient[j * layer_size + i];
        }
        
        // Update the bias
        biases[i] -= learning_rate * biases_gradient[i];
    }
} //----- end of updateWeights


void gradientDescent(float *weights, float *biases, float *error, float *layer_input, int input_size, int layer_size, float learning_rate) {
    // Accumulate the gradients
    float weights_gradient[input_size * layer_size] = {0};
    float biases_gradient[layer_size] = {0};
    accumulateGradient(weights_gradient, biases_gradient, error, layer_input, input_size, layer_size);

    // Update weights and biases
    updateWeights(weights, biases, weights_gradient, biases_gradient, input_size, layer_size, learning_rate);
    
} //----- end of gradientDescent


bool shuffleData(Data *data, int data_size) {
    random_device rd;
    mt19937 g(rd());
    shuffle(data, data + data_size, g);
    return true;
} //----- end of shuffleData