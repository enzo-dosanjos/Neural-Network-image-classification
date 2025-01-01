#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

using namespace std;
namespace fs = filesystem;

int main() {
    // Specify the directory to traverse
    string directory = "../mnist/testing"; // Change this to your target directory

    // Output CSV file
    ofstream csvFile("../mnist/testing_paths.csv");
    if (!csvFile) {
        cerr << "Failed to create the CSV file." << endl;
        return 1;
    }

    // Write the header row
    csvFile << "Full Path,Parent Directory" << endl;

    try {
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (fs::is_regular_file(entry)) { // Only include files
                string fullPath = entry.path().string();
                int parentDir = stoi(entry.path().parent_path().filename().string());

                // Write the file's full path and parent directory to the CSV file
                csvFile << '"' << fullPath << "\"," << parentDir << endl;
            }
        }
    } catch (const fs::filesystem_error& e) {
        cerr << "Filesystem error: " << e.what() << endl;
        return 1;
    }

    cout << "CSV file created successfully" << endl;
    csvFile.close();
    return 0;
}
