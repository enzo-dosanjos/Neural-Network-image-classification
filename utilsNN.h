/*************************************************************************
utilsNN - utility functions for neural network operations
                             -------------------
    copyright            : (C) 2024 by Enzo DOS ANJOS
*************************************************************************/

//---------- Interface of the module <utilsNN> (file utilsNN.h) -------------------
#if ! defined ( utilsNN_H )
#define utilsNN_H

//------------------------------------------------------------------------
// Role of the <utilsNN> module
// utility functions for neural network operations, including activation
// functions like ReLU and softmax, as well as functions for shuffling
// data and other mathematical operations.
//------------------------------------------------------------------------

/////////////////////////////////////////////////////////////////  INCLUDE
#include <iostream>
using namespace std;

#include "readFile.h"

//-------------------------------------------------------- Used interfaces

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//////////////////////////////////////////////////////////////////  PUBLIC
//------------------------------------------------------- Public functions

void ReLU (float *, int);
// Usage :
// Applies max(0, x) to each element of the array
// Contract :
//

void softmax (float *, float *, int);
// Usage :
// Applies exp(x) / sum(exp(x)) to each element of the array
// Contrat :
//

void mulMat(float *, int, float *, int, float *);
// Usage :
// Multiplies a matrix by a vector and stores the result in an array given in parameter
// Contrat :
//

void addMat(float *, float *, int, float *);
// Usage :
// Adds one vector to another and stores the result in an array given in parameter
// Contrat :
//

float crossEntropyLoss (float *, const int);
// Usage :
// Returns the cross-entropy loss for a probability distribution
// Contrat :
//

void backPropagation(float *, int, float *, float *, int, float *, string);
// Usage :
// Computes the error for the current layer based on the next layer's error and weights
// Contrat :
//

void accumulateGradient(float *, float *, float *, float *, int, int);
// Usage :
// Accumulates weight and bias gradients for a layer based on the current error and input values
// Contrat :
// Weight and bias gradient arrays must be initialized to 0

void updateWeights(float *, float *, float *, float *, int , int, float);
// Usage :
// Updates weights and biases using the calculated gradients and a given learning rate
// Contrat :
//

void gradientDescent(float *, float *, float *, float *, int , int , float);
// Usage :
// Combines gradient accumulation and weight updates to perform one step of gradient descent
// Contrat :
//

bool shuffleData(Data *, int);
// Usage :
// Randomizes the order of a dataset to improve model generalization during training
// Contrat :
//

#endif // utilsNN_H