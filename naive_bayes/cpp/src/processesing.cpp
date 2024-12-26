#include <sstream>
#include <vector>
#include <string>
#include "../include/processesing.h"
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



double compute_product(str_vec_t text, 
                       map_int_t histogram, 
                       int num_words, 
                       double proba, 
                       int alpha = 1) {
    double product = 1.0;

    for (const std::string& word : text) {
        // If the word is in the histogram, use its frequency. If not, apply Laplace smoothing.
        if (histogram.find(word) != histogram.end()) {
            product *= (static_cast<double>(histogram.at(word)) + alpha) / num_words;
        } else {
            product *= static_cast<double>(alpha) / num_words;
        }
    }

    return product * proba;
}

int vocabulary_counter(map_int_t data) {
  int sum {0};
    for (const auto& pair : data) {
      sum += pair.second;
    }
  return sum;
}


map_int_t build_histogram(vec_msv_t data) {
  map_int_t histogramMap;
  // iterate over vector 
  for (map_str_vec_t dict : data) {
      for (const auto& entry  : dict){ // for each word (element in vector) 
          const std::vector<std::string>& value = entry.second;
          for (const auto& word : value) {
            histogramMap[removePunctuation(lower_str(word))]++;

          }
      }     

  }
  return histogramMap;
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
