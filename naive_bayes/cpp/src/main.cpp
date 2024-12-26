#include <iostream>
#include <vector>
#include "../include/utils.h"
#include <unordered_map>

using VectorOfSVMs = std::vector<StringVectorMap>; // Vector of StringVectorMaps

int main () {
  std::string projectRoot {getProjectRoot()}; // Get root path of project
  std::string filename {projectRoot + "/data/emails/email.csv"}; // filename = root + path to data
                                                                 //
  VectorOfSVMs  data; // Create map
  data = parseCSV(filename); // Read csv file into array<unordered_map<string, array<string>> 
                                                                          //
  VectorOfSVMs train_data, test_data; // Initialize train and test datasets
  std::tie(train_data, test_data) = train_test_split(data, 0.8); // Split data
  
  // Get probabilities of ham and spam in training.
  int len_ham {0};
    for (const auto& dict : train_data) {
        if (dict.find("ham") != dict.end()) {
            len_ham++;
        }
    }

  double ham_proba {static_cast<double>(len_ham) / train_data.size()};
  double spam_proba {1.0 - ham_proba};
  // Separate ham and spam data from training.
  VectorOfSVMs ham_data, spam_data;
  
  for (const auto& dict : train_data) {
    if (dict.find("ham") != dict.end()) {
      ham_data.push_back({{"ham", dict.at("ham")}});
    }
    if (dict.find("spam") != dict.end()) {
      spam_data.push_back({{"spam", dict.at("spam")}});
    }
  }

  // Create histograms for both ham and spam in training
  std::unordered_map<std::string, int> ham_histogram, spam_histogram; // Create histogram map
  ham_histogram = build_histogram(ham_data);
  spam_histogram = build_histogram(spam_data);
  
  int num_ham_words {vocabulary_counter(ham_histogram)};
  int num_spam_words {vocabulary_counter(spam_histogram)};
      
  std::vector<std::string> y_hat, y;

  // Evaluation
  int alpha {1};
  for (const auto& email : test_data) {
    for (const auto& pair : email) {
      y.push_back(pair.first);
      double product_ham {compute_product(pair.second, ham_histogram, num_ham_words, ham_proba, alpha)};
      double product_spam {compute_product(pair.second, spam_histogram, num_spam_words, spam_proba, alpha)};
      y_hat.push_back(product_spam > product_ham ? "spam" : "ham");
    }
  }
  int correct {0};
  for (size_t i {0}; i < y_hat.size(); i++){
    if (y_hat[i] == y[i]) {
      correct++;
    }
  }
  std::cout << "Accuracy: " << (double) correct / test_data.size() << "%" << std::endl;
  return 0;
}
