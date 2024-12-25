#ifndef UTILS_H
#define UTILS_H
#include <string>
#include <vector>
#include <unordered_map>

using StringVector = std::vector<std::string>;
using StringVectorMap = std::unordered_map<std::string, StringVector>;
using VectorOfSVMs = std::vector<StringVectorMap>; // Vector of StringVectorMaps


std::string lower_str(std::string str);
std::vector<std::string> split(std::string &s, char delim);    
std::string removePunctuation(const std::string& word);
VectorOfSVMs parseCSV(const std::string& filename);
std::string getProjectRoot(); 
std::unordered_map<std::string, int> build_histogram(VectorOfSVMs data);
std::pair<VectorOfSVMs, VectorOfSVMs> train_test_split(VectorOfSVMs data, double ratio); 
int vocabulary_counter(std::unordered_map<std::string, int> data);
double compute_product(StringVector words, std::unordered_map<std::string, int> dict, int num_words, double proba, int alpha); 
#endif
