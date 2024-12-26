#ifndef UTILS_H
#define UTILS_H
#include <string>
#include <vector>
#include <unordered_map>

using str_vec_t = std::vector<std::string>;
using map_str_vec_t = std::unordered_map<std::string, str_vec_t>;
using vec_msv_t = std::vector<map_str_vec_t>; // Vector of map_str_vec_t
using map_int_t = std::unordered_map<std::string, int>;


str_vec_t split(std::string &s, char delim);    
vec_msv_t parseCSV(const std::string& filename);
std::string getProjectRoot(); 
std::pair<vec_msv_t, vec_msv_t> train_test_split(vec_msv_t data, double ratio); 
#endif
