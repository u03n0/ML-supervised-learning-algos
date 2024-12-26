#include <sstream>
#include <vector>
#include <string>
#include "../include/utils.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept> // std::runtime_error
#include <unordered_map>
#include <iostream>
#include <filesystem>
#include <random>
#include <utility>

using str_vec_t = std::vector<std::string>;
using map_str_vec_t = std::unordered_map<std::string, str_vec_t>;
using vec_msv_t = std::vector<map_str_vec_t>; // Vector of map_str_vec_t
using map_int_t = std::unordered_map<std::string, int>;



std::pair<vec_msv_t, vec_msv_t> train_test_split(vec_msv_t data, double ratio) {
  std::random_device rd;  // Get a random seed from the hardware (if available)
  std::default_random_engine rng(rd());  // Default random engine seeded with random_device
  std::shuffle(data.begin(), data.end(), rng);
  
  double split_index  { ratio * data.size()};
  vec_msv_t train_data {data.begin(), data.begin() + (int) split_index};
  vec_msv_t test_data {data.begin() + (int) split_index, data.end()};
  return std::make_pair(train_data, test_data);
}

std::vector<std::string> split(std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss (s);
  std::string item;

  while (getline (ss, item, delim)) {
    result.push_back (item);
  }
  return result;
}


std::string getProjectRoot() {
    // Start from the current working directory (cwd)
    std::filesystem::path cwd {std::filesystem::current_path()};

    // Walk up the directory tree until we find a file or folder that indicates the root (like .git or README)
    while (cwd != cwd.root_path()) {
        if (std::filesystem::exists(cwd / ".git") || std::filesystem::exists(cwd / "README.md")) {
            return cwd.string();  // Return the root path
        }
        cwd = cwd.parent_path();  // Move up one directory level
    }

    throw std::runtime_error("Project root not found! Ensure you're inside a Git repo or that the project structure is correct.");
}


std::string cleanWord(const std::string& word) {
    std::string cleanedWord;

    // Iterate over each character in the word
    for (char ch : word) {
        // Keep only alphabetic characters
        if (std::isalpha(static_cast<unsigned char>(ch))) {
            // Convert to lowercase and add to the result
            cleanedWord += std::tolower(static_cast<unsigned char>(ch));
        }
    }

    return cleanedWord;
}

// Function to parse the CSV file
vec_msv_t parseCSV(const std::string& filename) {
    // Open the CSV file
    std::ifstream file(filename);
    
    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        exit(1); // Exit if the file can't be opened
    }
    
    // Vector to store unordered_maps
    vec_msv_t data;
    
    std::string line;
    
    // Read each line from the CSV file
    while (getline(file, line)) {
        std::string category, message;
        
        // Find the position of the comma separating Category and Message
        size_t comma_pos = line.find(",");
        
        if (comma_pos != std::string::npos) {
            category = line.substr(0, comma_pos);
            message = line.substr(comma_pos + 1); // Everything after the comma
            
            // Remove leading/trailing whitespaces in category (optional)
            size_t start = category.find_first_not_of(" \t");
            size_t end = category.find_last_not_of(" \t");
            category = category.substr(start, end - start + 1);
            
            // Remove leading/trailing whitespaces in message (optional)
            start = message.find_first_not_of(" \t");
            end = message.find_last_not_of(" \t");
            message = message.substr(start, end - start + 1);
            
            // Split message into words
            std::vector<std::string> words;
            std::stringstream ss(message);
            std::string word;
            
            while (ss >> word) {
                // Clean the word: remove punctuation, remove numbers, and convert to lowercase
                word = cleanWord(word);

                // Add the word to the list if it's not empty
                if (!word.empty()) {
                    words.push_back(word);
                }
            }
            
            // Create a new unordered_map for the category and its associated words
            map_str_vec_t category_map;
            category_map[category] = words;
            
            // Add the unordered_map to the vector
            data.push_back(category_map);
        }
    }
    
    // Close the file
    file.close();
    
    return data;
}
