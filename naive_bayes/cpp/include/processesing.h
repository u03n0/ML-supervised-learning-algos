#ifndef PROCESSESING_H
#define PROCESSESING_H
#include <string>
#include <vector>
#include <unordered_map>

using str_vec_t = std::vector<std::string>;
using map_str_vec_t = std::unordered_map<std::string, str_vec_t>;
using vec_msv_t = std::vector<map_str_vec_t>; // Vector of map_str_vec_t
using map_int_t = std::unordered_map<std::string, int>;

std::string lower_str(std::string str);
std::string removePunctuation(const std::string& word);
map_int_t build_histogram(vec_msv_t data);
int vocabulary_counter(map_int_t data);
double compute_product(str_vec_t words, map_int_t dict, int num_words, double proba, int alpha); 
#endif
