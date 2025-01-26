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
#include "NN.h"

// readImg
#define STB_IMAGE_IMPLEMENTATION
#include "utilities/stb_image.h"

// readData
#include <fstream>
#include <sstream>

// Normalize
#include <opencv2/opencv.hpp>
#include <algorithm>
using namespace cv;

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


vector<Rect> mergeOverlappingRects(const vector<Rect>& rects, float overlapThreshold = 0.5, int proximityThreshold = 5) {
    vector<Rect> mergedRects;
    vector<bool> merged(rects.size(), false);  // avoid verifying two times the same rect

    for (size_t i = 0; i < rects.size(); ++i) {
        if (merged[i]) continue;

        Rect combinedRect = rects[i];  // Initialise with the current rect

        for (size_t j = i + 1; j < rects.size(); ++j) {
            if (merged[j]) continue;

            // Vérifier l'intersection ou la proximité
            Rect intersection = combinedRect & rects[j];
            float intersectionArea = (float)intersection.area();
            float unionArea = (float)(combinedRect.area() + rects[j].area() - intersectionArea);
            float overlap = intersectionArea / unionArea;

            // Verify if rectangles are close or overlapping
            if (overlap >= overlapThreshold || 
                (abs(combinedRect.x - rects[j].x) <= proximityThreshold && abs(combinedRect.y - rects[j].y) <= proximityThreshold)) {
                combinedRect |= rects[j];  // merge
                merged[j] = true;  // put it as merged
            }
        }

        mergedRects.push_back(combinedRect);
    }
    
    return mergedRects;
}


vector<Rect> filterSmallRects(const vector<Rect>& rects, int minArea) {
    vector<Rect> filteredRects;
    for (const auto& rect : rects) {
        if (rect.area() >= minArea) {  // Only keep rectangles with area >= minArea
            filteredRects.push_back(rect);
        }
    }
    return filteredRects;
}


bool Normalize(float *&pixArr, vector<float *> &roisArr, int width, int height, int newWidth, int newHeight, int nb_channels, bool multiple) {
    //  Create opencv image object
    Mat matImage;

    // Convert to grayscale if needed
    if (nb_channels == 3 || nb_channels == 4) {
        cout << "Converting to grayscale" << endl;
        // Create opencv image object
        matImage = Mat(height, width, nb_channels == 3 ? CV_32FC3 : CV_32FC4, pixArr);

        // Convert
        Mat grayImage;
        cvtColor(matImage, grayImage, COLOR_BGR2GRAY);

        // Update pixArr
        delete[] pixArr;
        pixArr = new float[width * height];
        memcpy(pixArr, grayImage.ptr<float>(), width * height * sizeof(float));
        matImage = grayImage;

    } else {
        matImage = Mat(height, width, CV_32FC1, pixArr);
    }

    if (multiple) {
        // Detect the regions of interest in the image
        // Apply Gaussian Blur with a 5x5 kernel
        if (width * height >= 400*400) {
            GaussianBlur(matImage, matImage, Size(3, 3), 0);
        }

        // Convert to CV_8UC1 for Canny
        Mat detContImage8U;
        matImage.convertTo(detContImage8U, CV_8UC1, 255.0);

        // Apply Canny to detect Contours
        Mat edges;
        Canny(detContImage8U, edges, 30, 150);

        // Find Contours
        vector<vector<Point>> contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Sort contours by their x position
        sort(contours.begin(), contours.end(),
                [](const vector<Point>& c1, const vector<Point>& c2) {
                    return boundingRect(c1).x < boundingRect(c2).x;
                });

        // Get bounding rectangles
        vector<Rect> rects;
        for (const auto &contour : contours) {
            // Get bounding rectangle
            Rect rect = boundingRect(contour);
            rects.push_back(rect);
        }


        vector<Rect> mergedRects = mergeOverlappingRects(rects, 0.05, 10.0);

        vector<Rect> finalRects = filterSmallRects(mergedRects, width > height ? width : height);


        // Crop ROIs and process
        int i = 0;
        for (const auto &rect : finalRects) {
            // Crop
            Mat roi = matImage(rect);

            // Add a border to the number so it does not touch the edges of the image
            copyMakeBorder(roi, roi, 5, 5, 5, 5, BORDER_CONSTANT, roi.data[0]);
            
            // Resize the image
            resize(roi, roi, Size(newWidth, newHeight), 0, 0, INTER_LINEAR);


            // Normalize the pixel values (divide by 255)
            roi.convertTo(roi, CV_32FC1, 1.0 / 255.0);


            // Calculate mean intensity to determine if the background is white
            float mean_intensity = mean(roi)[0];

            if (mean_intensity > 0.5f) { // Invert if most pixels are white
                roi = Scalar(1.0) - roi;
            }


            // Convert Mat to float* and store in roisArr
            float *roiArr = new float[newWidth * newHeight];
            memcpy(roiArr, roi.ptr<float>(), newWidth * newHeight * sizeof(float));
            roisArr.push_back(roiArr);
        }

    } else {
        // Resize the image
        resize(matImage, matImage, Size(newWidth, newHeight), 0, 0, INTER_LINEAR);


        // Normalize the pixel values (divide by 255)
        matImage.convertTo(matImage, CV_32FC1, 1.0 / 255.0);


        // Calculate mean intensity to determine if the background is white
        float mean_intensity = mean(matImage)[0];

        if (mean_intensity > 0.5f) { // Invert if most pixels are white
            matImage = Scalar(1.0) - matImage;
        }


        // Convert Mat to float* and store in roisArr
        float *pixArr = new float[newWidth * newHeight];
        memcpy(pixArr, matImage.ptr<float>(), newWidth * newHeight * sizeof(float));
        roisArr.push_back(pixArr);
    }

    return true;
} //----- end of Normalize


bool readData (string csv_path, Data *data, int batch_size) {
    // Input CSV file
    ifstream csvFile(csv_path);
    if (!csvFile) {
        cerr << "Failed to open the CSV file." << endl;
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


void saveModel (const string filename, vector<Layer> &NN) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Unable to open file for saving model." << endl;
        return;
    }

    // Save the number of layers
    int num_layers = NN.size();
    file.write(reinterpret_cast<char*>(&num_layers), sizeof(int));

    // Save each layer
    for (int i = 0; i < num_layers; i++) {
        Layer layer = NN[i];

        // Save the layer type
        int type_length = layer.type.length();
        file.write(reinterpret_cast<char*>(&type_length), sizeof(int));
        file.write(layer.type.c_str(), type_length);

        // Save the layer input and output sizes
        file.write(reinterpret_cast<char*>(&layer.input_size), sizeof(int));
        file.write(reinterpret_cast<char*>(&layer.output_size), sizeof(int));

        // Save the activation function parameters
        int num_params = layer.activation_func_params.size();
        file.write(reinterpret_cast<char*>(&num_params), sizeof(int));
        if (num_params > 0) {
            file.write(reinterpret_cast<char*>(layer.activation_func_params.data()), num_params * sizeof(float));
        }

        // Save the layer weights and biases
        file.write(reinterpret_cast<char*>(layer.weights), layer.input_size * layer.output_size * sizeof(float));
        file.write(reinterpret_cast<char*>(layer.biases), layer.output_size * sizeof(float));
    }

    file.close();
    cout << "Model saved to " << filename << endl;
} //----- end of saveModel


void loadModel(const string filename, vector<Layer> &NN) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Unable to open file for loading model." << endl;
        return;
    }

    // destroy the current model if needed
    if (NN.size() > 0) {
        destroyNN(NN);
    }

    // Load the number of layers
    int num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));

    // Load each layer
    for (int i = 0; i < num_layers; i++) {
        // Load the type
        int type_length;
        file.read(reinterpret_cast<char*>(&type_length), sizeof(int));

        char *type = new char[type_length + 1];
        file.read(type, type_length);
        type[type_length] = '\0';  // Null-terminate the string
        string layer_type = type;
        delete[] type;

        // Load the input and output sizes
        int input_size, output_size;
        file.read(reinterpret_cast<char*>(&input_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&output_size), sizeof(int));

        // Load the activation function parameters
        int num_params;
        file.read(reinterpret_cast<char*>(&num_params), sizeof(int));
        vector<float> activation_func_params(num_params);
        if (num_params > 0) {
            file.read(reinterpret_cast<char*>(activation_func_params.data()), num_params * sizeof(float));
        }

        // Add the layer to the model
        addLayer(NN, layer_type, output_size, input_size, activation_func_params);

        // Load the weights and biases
        file.read(reinterpret_cast<char*>(NN.back().weights), input_size * output_size * sizeof(float));
        file.read(reinterpret_cast<char*>(NN.back().biases), output_size * sizeof(float));
    }

    file.close();
    cout << "Model loaded from " << filename << endl;
} //----- end of loadModel