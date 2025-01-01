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

//-------------------------------------------------------- Used interfaces

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//////////////////////////////////////////////////////////////////  PUBLIC
//------------------------------------------------------- Public functions

void initLayer (float *, float *, int, int);
// Usage :
// Initializes the weights and biases of a layer with random values for 
// weights and zero for biases
// Contract :
//

void computeLayer (float *, float *, float *, int, int, float *, string);
// Usage :
// Computes the output of a layer by applying the weights and biases to 
// the input, and then applying the activation function given in parameter
// Contract :
//

void computeError (float *, int, float *);
// Usage :
// Calculates the error between the predicted output and the true label.
// Contract :
//

void trainModel (string, string, int, int, int, int, float);
// Usage :
// Trains the model using a given dataset. The paths and labels of the
// dataset are loaded in an array which is shuffled. Then, the NN accumulate
// gradients on batch of given size and update weights at the end of the
// batch to save computational time. Finally, the model is saved to a binary file.
// Contract :
//

void improveModel (string, string, int, int, int, int, float);
// Usage :
// same as trainModel but the model is loaded from a binary file
// Contract :
//

void testModel (string, string, int, int);
// Usage :
// Tests the model using a given dataset. Gives the accuracy of the model
// at the end of the test
// Contract :
// data_size ne doit pas être nul

void predict (string, string, int);
// Usage :
// Uses a given model to predict the label of an image
// Contract :
//

#endif // NN_H