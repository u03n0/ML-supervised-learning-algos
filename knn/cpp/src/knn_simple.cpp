#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include "../../../utils/cpp_utils.h"
#include "../include/utils.h"


using str_vec_t = std::vector<std::string>;
using map_str_vec_t = std::unordered_map<std::string, str_vec_t>;
using vec_msv_t = std::vector<map_str_vec_t>; // Vector of map_str_vec_t
using map_int_t = std::unordered_map<std::string, int>;




int main () {
    std::string projectRoot {getProjectRoot()}; // Get root path of project
    std::string filename {projectRoot + "/data/emails/email.csv"}; // filename = root + path to data
    vec_msv_t  data; // Create map
    data = parseCSV(filename); // Read csv file into array<unordered_map<string, array<string>> 
    vec_msv_t train_data, test_data; // Initialize train and test datasets
    std::tie(train_data, test_data) = train_test_split(data, 0.8); // Split data
    
    for (const auto& item: train_data) {
        for (const auto& pair: item) {
            std::cout << pair.first << std::endl;
        }
    }

        return 0;
    }