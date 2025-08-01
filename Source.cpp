#include "RandomForest.h"
#include <fstream>
#include <iomanip>

// function to load passengers csv data
std::vector<Passenger> loadData(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "File not found\n";
    exit(1);
  }
  std::string line;
  std::vector<Passenger> data;

  std::vector<std::string> fields;

  std::getline(file, line); // skip header
  while (std::getline(file, line)) {
    bool inQuotes = false;
    std::string temp = "";
    for (char c : line) {
      if (c == '"')
        inQuotes = !inQuotes;
      else if (c == ',' && !inQuotes) {
        fields.push_back(temp);
        temp.clear();
        temp = "";
      } else
        temp += c;
    }
    fields.push_back(temp);
    if (fields.size() >= 12)
      data.emplace_back(fields);
    fields.clear();
  }
  return data;
}

int main() {
    std::vector<Passenger> data{ loadData("titanic.csv") };
    std::string tree_model_file = R"(C:\Users\HP\source\repos\decisiontree\decisiontree\tree_model.bin)";
    std::string forest_model_file = R"(C:\Users\HP\source\repos\decisiontree\decisiontree\forest_model.bin)";

    // Split into train and test
    std::random_device rd;
    std::mt19937 mt(rd());
    std::shuffle(std::begin(data), std::end(data), mt);

    int splitPoint = (int)(data.size() * .8);
    std::vector<Passenger> trainData(std::begin(data),
                                    std::begin(data) + splitPoint);
    std::vector<Passenger> testData(std::begin(data) + splitPoint,
                                    std::end(data));

    // Train and evaluate a single decision tree
    DecisionTree tree(5, 2, 2);
    tree.train(trainData);

    int correct = 0;
    for (const auto& p : testData) {
        if (tree.predict(p) == p.survived)
            correct++;
    }
    std::cout << "Decision Tree Accuracy: "
        << (double)(correct / (double)testData.size()) << "\n";

    tree.save(tree_model_file);


    DecisionTree tree1;
    tree1.load(tree_model_file);

    correct = 0;
    for (const auto& p : testData) {
        if (tree1.predict(p) == p.survived)
            correct++;
    }
    std::cout << "Decision Tree Accuracy: "
        << (double)(correct / (double)testData.size()) << "\n";

  
    RandomForest forest(100, 5, 2, 2, 0.7);
    forest.train(trainData);

    auto importance = forest.computeFeatureImportances();
    for (const auto &[feature, score] : importance) {
    std::cout << "Feature " << feature << ": " << std::fixed
                << std::setprecision(4) << score << '\n';
    }

    double randomForestAccuracy = forest.evaluate(testData);
    std::cout << "Random Forest Accuracy: " << randomForestAccuracy << std::endl;

    forest.save(forest_model_file);

    RandomForest forest1;
    forest1.load(forest_model_file);

    randomForestAccuracy = forest1.evaluate(testData);
    std::cout << "Random Forest Accuracy: " << randomForestAccuracy << std::endl;

    return 0;
}