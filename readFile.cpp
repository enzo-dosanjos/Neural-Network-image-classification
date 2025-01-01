/*************************************************************************
readFile - reads or writes files (images, data, models) for the neural network
                             -------------------
    copyright            : (C) 2024 by Enzo DOS ANJOS
*************************************************************************/

//---------- Implementation of the module <readFile> (file readFile.cpp) ---------------

/////////////////////////////////////////////////////////////////  INCLUDE
//--------------------------------------------------------- System Include
#include <iostream>
using namespace std;

//------------------------------------------------------- Personal Include
#include "readFile.h"

// readImg
#define STB_IMAGE_IMPLEMENTATION
#include "utilities/stb_image.h"

// readData
#include <fstream>
#include <sstream>

/////////////////////////////////////////////////////////////////  PRIVATE
//-------------------------------------------------------------- Constants

//------------------------------------------------------------------ Types

//------------------------------------------------------- Static Variables

//------------------------------------------------------ Private Functions

//////////////////////////////////////////////////////////////////  PUBLIC
//------------------------------------------------------- Public Functions
bool readImg (const string filename, float *&pixArr, int &width, int &height, int &channels)
{   
    // Load the image file
    unsigned char *temp = nullptr;
    temp = stbi_load(filename.c_str(), &width, &height, &channels, 0);

    if (temp == nullptr) {
        cerr << "Error: Could not load the image file" << endl;
        return false;
    }

    // copy the pixel values to the array
    pixArr = new float[width * height * channels];
    for (int i = 0; i < width * height * channels; i++) {
        pixArr[i] = static_cast<float>(temp[i]);
    }

    // Free the image memory when done
    stbi_image_free(temp);

    return true;

} //----- end of readImg


bool Normalize(float *&pixArr, int width, int height, int newWidth, int newHeight, int nb_channels) {
    // Convert to grayscale if needed
    if (nb_channels == 3 || nb_channels == 4) {
        cout << "Converting to grayscale" << endl;
        float *greyPixArr = new float[width * height];
        for (int i = 0; i < width * height; i++) {
            // Use only the RGB channels (ignore the alpha channel for pngs)
            greyPixArr[i] = 0.299f * pixArr[i * nb_channels] +
                            0.587f * pixArr[i * nb_channels + 1] +
                            0.114f * pixArr[i * nb_channels + 2];
        }
        delete[] pixArr;
        pixArr = greyPixArr;
    }

    // Resize the image if needed
    if (width != newWidth || height != newHeight) {
        cout << "Resizing image" << endl;
        float *resizedPixArr = new float[newWidth * newHeight];
        float x_ratio = static_cast<float>(width) / newWidth;
        float y_ratio = static_cast<float>(height) / newHeight;
        float px, py;
        for (int i = 0; i < newHeight; i++) {
            for (int j = 0; j < newWidth; j++) {
                px = floor(j * x_ratio);
                py = floor(i * y_ratio);
                resizedPixArr[(i * newWidth) + j] = pixArr[(static_cast<int>(py) * width) + static_cast<int>(px)];
            }
        }
        delete[] pixArr;
        pixArr = resizedPixArr;
    }

    // Normalize the pixel values (divide by 255)
    int total_pixels = newWidth * newHeight;
    float total_intensity = 0.0f;

    for (int i = 0; i < total_pixels; i++) {
        pixArr[i] = pixArr[i] / 255.0f;
        total_intensity += pixArr[i];
    }

    // Calculate mean intensity to determine if the background is white
    float mean_intensity = total_intensity / total_pixels;

    if (mean_intensity > 0.5f) { // Invert if most pixels are white
        for (int i = 0; i < total_pixels; i++) {
            pixArr[i] = 1.0f - pixArr[i];
        }
    }

    return true;
} //----- end of Normalize


bool readData (string csv_path, Data *data, int batch_size) {
    // Input CSV file
    ifstream csvFile(csv_path);
    if (!csvFile) {
        cerr << "Failed to open the CSV file." << std::endl;
        return false;
    }

    string line;

    getline(csvFile, line); // Skip the header row

    // Read data into the array
    int index = 0;
    while (getline(csvFile, line) && index < batch_size) {
        istringstream lineStream(line);
        string path, label_str;

        // Parse the CSV line
        if (getline(lineStream, path, ',') && getline(lineStream, label_str, ',')) {
            // Remove quotes if present
            if (!path.empty() && path.front() == '"' && path.back() == '"') {
                path = path.substr(1, path.size() - 2);
            }
            if (!label_str.empty() && label_str.front() == '"' && label_str.back() == '"') {
                label_str = stoi(label_str.substr(1, label_str.size() - 2));
            }

            int label = stoi(label_str);  //convert the label to an integer

            // Add the data to the array
            data[index++] = {path, label};
        }
    }

    csvFile.close();
    return true;
} //----- end of readData

void saveModel(const string filename, int input_size, float *layer_1_weight, float *layer_1_bias, int layer_1_size, 
                                                        float *output_weight, float *output_bias, int output_size) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Unable to open file for saving model." << endl;
        return;
    }

    // Save layer 1 weights and biases
    file.write(reinterpret_cast<char*>(layer_1_weight), input_size * layer_1_size * sizeof(float));
    file.write(reinterpret_cast<char*>(layer_1_bias), layer_1_size * sizeof(float));

    // Save output layer weights and biases
    file.write(reinterpret_cast<char*>(output_weight), layer_1_size * output_size * sizeof(float));
    file.write(reinterpret_cast<char*>(output_bias), output_size * sizeof(float));

    file.close();
    cout << "Model saved to " << filename << endl;
} //----- end of saveModel


void loadModel(const string filename, int input_size, float *layer_1_weight, float *layer_1_bias, int layer_1_size,
                                                        float *output_weight, float *output_bias, int output_size) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Unable to open file for loading model." << endl;
        return;
    }

    // Load layer 1 weights and biases
    file.read(reinterpret_cast<char*>(layer_1_weight), input_size * layer_1_size * sizeof(float));
    file.read(reinterpret_cast<char*>(layer_1_bias), layer_1_size * sizeof(float));

    // Load output layer weights and biases
    file.read(reinterpret_cast<char*>(output_weight), layer_1_size * output_size * sizeof(float));
    file.read(reinterpret_cast<char*>(output_bias), output_size * sizeof(float));

    file.close();
    cout << "Model loaded from " << filename << endl;
} //----- end of loadModel