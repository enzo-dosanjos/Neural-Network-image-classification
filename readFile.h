/*************************************************************************
readFile - reads or writes files (images, data, models) for the neural network
                             -------------------
    copyright            : (C) 2024 by Enzo DOS ANJOS
*************************************************************************/

//---------- Interface of the module <readFile> (file readFile.h) -------------------
#if ! defined ( readFile_H )
#define readFile_H

//------------------------------------------------------------------------
// Role of the <readFile> module
// utility functions for loading and processing file contents, such as images 
// or CSVs, into memory. Also includes functions for saving and loading models.
// saved in binary format.
//------------------------------------------------------------------------

/////////////////////////////////////////////////////////////////  INCLUDE
#include <iostream>
using namespace std;

#include <vector>

//-------------------------------------------------------- Used interfaces
#include "NN.h"

//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types
typedef struct {
    string path;
    int label;
} Data;

//////////////////////////////////////////////////////////////////  PUBLIC
//------------------------------------------------------- Public functions
bool readImg (const string, float *&, int &, int &, int &);
// Usage :
// Loads an image (Jpeg, PNG, ...) file into an array and it's dimensions 
// into given parameters
// Contract :
//

bool Normalize(float *&, int, int, int, int, int);
// Usage :
// Converts a pixel array to grayscale, resizes it, invert it (if needed) 
// and normalizes the values between 0 and 1
// Contract :
//

bool readData (string, Data *, int);
// Usage :
// Reads a CSV file and extracts paths and labels into a Data array
// Contract :
//

void saveModel(const string, vector<Layer> &);
// Usage :
// Saves the number of layers and the layer's weights, biases, type 
// and sizes into a binary file
// Contract :
//

void loadModel(const string, vector<Layer> &);
// Usage :
// Loads the number of layers and the layer's weights, biases, type 
// and sizes from a binary file into a vector of layers given in parameter
// Contract :
//

#endif // readFile_H