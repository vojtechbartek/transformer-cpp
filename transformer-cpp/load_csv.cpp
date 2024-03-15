#include <fstream>
#include <string>
#include <sstream>
#include <vector>


std::vector<float> split(const std::string &s, char delimiter) {
    /*
    * Split a string by a delimiter and return a vector of floats
    * @param s: string to split
    * @param delimiter: delimiter to split by
    * @return: vector of floats
    */

    std::vector<float> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(std::stof(token));
    }
    return tokens;
}

std::vector<std::vector<float>> load_embeddings(const std::string &filename) {
    /*
    * Load a CSV file into a vector of vectors
    * @param filename: name of the file to load
    * @return: vector of vectors
    */

    std::ifstream file(filename); // Open the file
    std::vector<std::vector<float>> data; // Create a vector to store the data
    std::string line;

    while (std::getline(file, line)) {
        data.push_back(split(line, ',')); // Split the line and add to the data
    }
    file.close(); // Close the file
    return data;
}

std::vector<int> load_targets(const std::string &filename) {
    /*
    * Load a CSV file into a vector of integers
    * the file might have single line or multiple lines with 
    * integers separated by column and or new line
    * @param filename: name of the file to load
    * @return: vector of integers
    */

    std::ifstream file(filename); // Open the file
    std::vector<int> data; // Create a vector to store the data
    std::string line;

    while (std::getline(file, line)) {
        std::vector<float> tokens = split(line, ',');
        for (auto &token : tokens) {
            data.push_back((int)token);
        }
    }
    file.close(); // Close the file
    return data;
}

    
