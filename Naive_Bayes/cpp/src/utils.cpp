#include <sstream>
#include <vector>
#include <string>
#include "../include/utils.h"
#include <algorithm>
#include <cctype>


std::vector<std::string> split (std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss (s);
  std::string item;

  while (getline (ss, item, delim)) {
    result.push_back (item);
  }
  return result;
}

std::string lower_str(std::string str) {
  std::string lowered_str = "";
  for (char c: str){
    lowered_str += tolower(c);
  }
  return lowered_str;
}


std::string removePunctuation(const std::string& word) {
  std::string cleanedWord = word;
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

