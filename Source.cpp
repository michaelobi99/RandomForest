#include "RandomForest.h"
#include <fstream>



//function to load passengers csv data
std::vector<Passenger> loadData(const std::string& filename) {
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
			if (c == '"') inQuotes = !inQuotes;
			else if (c == ',' && !inQuotes) {
				fields.push_back(temp);
				temp.clear();
				temp = "";
			}
			else temp += c;
		}
		fields.push_back(temp);
		if (fields.size() >= 12) data.emplace_back(fields);
		fields.clear();
	}
	return data;
}

int main() {
	std::vector<Passenger> data{ loadData("titanic.csv") };

	//Split into train and test
	std::random_device rd;
	std::mt19937 mt(rd());
	std::shuffle(std::begin(data), std::end(data), mt);

	int splitPoint = (int)(data.size() * .8);
	std::vector<Passenger> trainData(std::begin(data), std::begin(data) + splitPoint);
	std::vector<Passenger> testData(std::begin(data) + splitPoint, std::end(data));

	//Train and evaluate a single decision tree
	DecisionTree tree(7, 3, 3);
	tree.train(trainData);
	int correct = 0;
	for (const auto& p : testData) {
		if (tree.predict(p) == p.survived) correct++;
	}
	std::cout << "Decision Tree Accuracy: " << (double)(correct / (double)testData.size()) << "\n";

	//Train and evaluate random forest
	RandomForest forest(100, 7, 3, 3, std::sqrt(trainData.size()));
	forest.train(trainData);
	double randomForestAccuracy = forest.evaluate(testData);
	std::cout << "Random Forest Accuracy: " << randomForestAccuracy << std::endl;
	return 0;

}