/*************************************************************************
NN - implement a basic neural network with one hidden layer for now
                             -------------------
    copyright            : (C) 2024 by Enzo DOS ANJOS
*************************************************************************/

//---------- Interface of the module <NN> (file NN.h) -------------------
#if ! defined ( NN_H )
#define NN_H

//------------------------------------------------------------------------
// Role of the <NN> module
// This module provides functions to initialize layers, compute layer
// outputs and error, train/ improve a model, test a model, and make 
// predictions using a simple neural network with one hidden layer for now.
// The neural network is designed for image classification tasks using 
// backpropagation and gradient descent optimization.
//------------------------------------------------------------------------

/////////////////////////////////////////////////////////////////  INCLUDE
#include <iostream>
using namespace std;

// store the NN layers
#include <vector>

//-------------------------------------------------------- Used interfaces

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types
struct Layer {
    string type;                // "softmax", "ReLU"...
    vector<float> activation_func_params;  // parameters for the activation function like alpha for ELU
    int input_size;
    int output_size;

    float *weights;
    float *biases;
    float *output;

    float *weight_gradient;
    float *bias_gradient;
    float *error;
};

//////////////////////////////////////////////////////////////////  PUBLIC
//------------------------------------------------------- Public functions

void initLayer (float *, float *, int, int);
// Usage :
// Initializes the weights and biases of a layer with random values for 
// weights and zero for biases
// Contract :
//

void initLayerGradient (float *, float *, int, int);
// Usage :
// Initializes the weights and biases' gradients of a layer to zero
// Contract :
//

void computeLayer (float *, float *, float *, int, int, float *, string, const vector<float> &);
// Usage :
// Computes the output of a layer by applying the weights and biases to 
// the input, and then applying the activation function given in parameter
// Contract :
//

void addLayer(vector<Layer> &, string, int, int, const vector<float> &);
void addLayer(vector<Layer> &, string, int, int);
void addLayer(vector<Layer> &, string, int, const vector<float> &);
void addLayer(vector<Layer> &, string, int);
// Usage :
// adds a layer to the neural network which is stored in a vector of layers with
// the given type, input size (not needed if a layer already exists) and output size
// Contract :
//

void computeError (float *, int, float *);
// Usage :
// Calculates the error between the predicted output and the true label.
// Contract :
//

void destroyNN(vector<Layer> &);
// Usage :
// Fully destroys the neural network by freeing the memory allocated for
// the weights, biases and output arrays of each layer and clear the vector
// Contract :
//

void trainModel (vector<Layer> &, string, string, int, int, int, float);
// Usage :
// Trains the model using a given dataset. The paths and labels of the
// dataset are loaded in an array which is shuffled. Then, the NN accumulate
// gradients on batch of given size and update weights at the end of the
// batch to save computational time. Finally, the model is saved to a binary file.
// Contract :
//

void improveModel (vector<Layer> &, string, string, int, int, int, float);
// Usage :
// same as trainModel but the model is loaded from a binary file
// Contract :
//

int predictLabel (vector<Layer> &, string, float &);
// Usage :
// Predicts the label of an image using a given model
// Contract :
//

void testModel (vector<Layer> &, string, string, int);
// Usage :
// Tests the model using a given dataset. Gives the accuracy of the model
// at the end of the test
// Contract :
// data_size ne doit pas être nul

void predict (vector<Layer> &, string, string);
// Usage :
// Uses a given model loaded from a binary file to predict the label of 
// a given image
// Contract :
//

#endif // NN_H