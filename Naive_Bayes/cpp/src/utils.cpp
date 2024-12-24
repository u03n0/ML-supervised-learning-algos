#include <sstream>
#include <vector>
#include <string>
#include "../include/utils.h"

std::vector<std::string> split (std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss (s);
  std::string item;

  while (getline (ss, item, delim)) {
    result.push_back (item);
  }
  return result;
}

/*std::string lower_str(std::string str) {
  std::string lowered_str = "";
  for (char c: str){
    lowered_str += tolower(c);
  }
  return lowered_str;
}
*/

