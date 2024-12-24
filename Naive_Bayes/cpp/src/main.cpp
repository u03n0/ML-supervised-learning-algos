#include <iostream>
#include <vector>
#include "../include/utils.h"
#include <unordered_map>



int main () {
  std::vector<std::string> texts 
    = {"The cat plays with the red ball.", "I have a cat.", "The dog chases the cat."};
  std::unordered_map<std::string, int> myUnorderedMap;
 // iterate over vector 
  int i = 1;
  for (std::string elem : texts) {
    std::vector <std::string> chopped = split(elem, ' '); // create vector of strings with blank space as delimter.
      for (const std::string& word : chopped){ // for each word (element in vector) 
          myUnorderedMap[removePunctuation(lower_str(word))]++;
      }
  }
  for (const auto& pair : myUnorderedMap) {
    std::cout << pair.first << "->" << pair.second << std::endl;
  }
  return 0;
}
