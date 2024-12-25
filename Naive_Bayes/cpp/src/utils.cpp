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

using StringVector = std::vector<std::string>;
using StringVectorMap = std::unordered_map<std::string, StringVector>;
using VectorOfSVMs = std::vector<StringVectorMap>; // Vector of StringVectorMaps



double compute_product(StringVector words, std::unordered_map<std::string, int> dict, int num_words, double proba, int alpha) {
  double product {1.0};
  for (std::string word : words) {
      product *= dict.find(word) != dict.end()? (dict[word] + alpha) / num_words : alpha / num_words; 
    }
  return product * proba;
}

int vocabulary_counter(std::unordered_map<std::string, int> data) {
  int sum {0};
    for (const auto& pair : data) {
      sum += pair.second;
    }
  return sum;
}


std::pair<VectorOfSVMs, VectorOfSVMs> train_test_split(VectorOfSVMs data, double ratio) {
  std::random_device rd;  // Get a random seed from the hardware (if available)
  std::default_random_engine rng(rd());  // Default random engine seeded with random_device
  std::shuffle(data.begin(), data.end(), rng);
  
  double split_index  { ratio * data.size()};
  VectorOfSVMs train_data {data.begin(), data.begin() + (int) split_index};
  VectorOfSVMs test_data {data.begin() + (int) split_index, data.end()};
  return std::make_pair(train_data, test_data);
}


std::unordered_map<std::string, int> build_histogram(VectorOfSVMs data) {
  std::unordered_map<std::string, int> histogramMap;
  // iterate over vector 
  for (StringVectorMap dict : data) {
      for (const auto& entry  : dict){ // for each word (element in vector) 
          const std::vector<std::string>& value = entry.second;
          for (const auto& word : value) {
            histogramMap[removePunctuation(lower_str(word))]++;

          }
      }     

  }
  return histogramMap;
}


VectorOfSVMs parseCSV(const std::string& filename) {
    // Open the CSV file
  std::ifstream file(filename);
    // Check if the file was opened successfully
    if (!file.is_open()) {
      std::cerr << "Could not open the file!" << std::endl;
        exit(1); // Exit if the file can't be opened
    }
    
    // Vector to store unordered_maps
    VectorOfSVMs data;
    
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
                words.push_back(word);
            }
            
            // Create a new unordered_map for the category and its associated words
            StringVectorMap category_map;
            category_map[category] = words;
            
            // Add the unordered_map to the vector
            data.push_back(category_map);
        }
    }
    
    // Close the file
    file.close();
    
    return data;
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

std::string lower_str(std::string str) {
  std::string lowered_str {""};
  for (char c: str){
    lowered_str += tolower(c);        
  }
  return lowered_str;
}


std::string removePunctuation(const std::string& word) {
  std::string cleanedWord {word};
  cleanedWord.erase(
      cleanedWord.begin(),
      std::find_if(cleanedWord.begin(), cleanedWord.end(), [](unsigned char ch) { return std::isalnum(ch); })
    );
    
    cleanedWord.erase(
        std::find_if(cleanedWord.rbegin(), cleanedWord.rend(), [](unsigned char ch) { return std::isalnum(ch); }).base(),
        cleanedWord.end()
    );
    return cleanedWord;
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

