#include <iostream>
#include <vector>
#include "../include/utils.h"


int main () {

  std::vector<std::string> texts 
    = {"Michi went to the park today and saw a bird.", "I love my cat, hes perfect.",
    "Michi is a cat and he is mine.", "A cat is mans bestfriend, and Michi is my cat."
    };

  std::string str = "Test string..";
  std::vector<std::string> frag = split(str, ' ');
  for (std::string elem : frag) {
    std::cout << elem << std::endl;
  }
  return 0;
}
