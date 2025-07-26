#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <numeric>

//Passenger data structure
struct Passenger {
    int passengerID;
    bool survived;
    int pclass; //ticket class
    std::string name;
    std::string sex;
    int age;
    int sibSp; //sibling or spouse aboard
    int parch; //number of parent/children aboard
    std::string ticket;
    int fare;
    std::string cabin;
    std::string embarked; //port(C, Q, S)

    Passenger(const std::vector<std::string>& fields) {
        passengerID = std::stoi(fields[0]);
        survived = std::stoi(fields[1]) == 1;
        pclass = std::stoi(fields[2]);
        name = fields[3];
        sex = fields[4];
        try {
            age = fields[5].empty() ? -1 : std::stoi(fields[5]);
        }
        catch (...) {
            age = -1;
        }
        sibSp = std::stoi(fields[6]);
        parch = std::stoi(fields[7]);
        ticket = fields[8];
        try {
            fare = fields[9].empty() ? -1 : std::stoi(fields[9]);
        }
        catch (...) {
            fare = -1;
        }
        cabin = fields[10];
        embarked = fields[11].empty() ? "U" : fields[11].substr(0, 1);
    }
    friend std::ostream& operator<<(std::ostream& stream, const Passenger& p) {
        stream << p.passengerID << "," << p.survived << "," << p.pclass << "," << p.name << "," << p.sex << "," << p.age << "," << p.sibSp << "," << p.parch << "," << p.ticket << "," << p.fare << "," << p.cabin << "," << p.embarked << "\n";
        return stream;
    }
};


struct TreeNode {
    int featureIdx; //Feature index used for splitting (-1 for leaf)
    double splitValue; //split threshold for numerical features
    std::string splitCategory; //for categorical splits
    bool isLeaf;
    bool leafClass; //class prediction if leaf
    TreeNode* left;
    TreeNode* right;
    TreeNode() : featureIdx(-1), splitValue(0), isLeaf(false), leafClass(false), left(nullptr), right(nullptr) {}
    ~TreeNode() {
        if (left) delete left;
        if (right) delete right;
        left = right = nullptr;
    }
};

class DecisionTree {
private:
    TreeNode* root;
    int maxDepth;
    int minSamplesSplit;
    int minSamplesLeaf;
    double featureSampleRatio;
    std::unordered_map<int, double> featureImportance;

    //Calculate Gini impurity
    double calculateGini(const std::vector<bool>& labels) {
        if (labels.empty()) return 0.0;
        unsigned count0 = 0, count1 = 0;
        for (bool label : labels) {
            if (label) ++count1;
            else ++count0;
        }
        double p0 = (double)(count0) / labels.size();
        double p1 = (double)(count1) / labels.size();
        return 1.0 - (p0 * p0 + p1 * p1);
    }

    //Split dataset based on feature and value
    std::pair<std::vector<int>, std::vector<int>> splitData(const std::vector<Passenger>& data, const std::vector<int>& indices, int featureIdx, double splitValue, const std::string& splitCategory = "") {
        std::vector<int> leftIndices, rightIndices;
        for (int idx : indices) {
            const Passenger& p = data[idx];
            bool goLeft = false;

            if (featureIdx == 0) { // pclass
                goLeft = p.pclass <= splitValue;
            }
            else if (featureIdx == 1) { // sex
                goLeft = (splitCategory.empty()) ? (p.sex == "female") : (p.sex == splitCategory);
            }
            else if (featureIdx == 2) { // age
                goLeft = p.age <= splitValue && p.age >= 0;
            }
            else if (featureIdx == 3) { // sibSp
                goLeft = p.sibSp <= splitValue;
            }
            else if (featureIdx == 4) { // parch
                goLeft = p.parch <= splitValue;
            }
            else if (featureIdx == 5) { // fare
                goLeft = p.fare <= splitValue && p.fare >= 0;
            }
            else if (featureIdx == 6) { // embarked
                goLeft = (splitCategory.empty()) ? (p.embarked == "C") : (p.embarked == splitCategory);
            }

            if (goLeft) leftIndices.push_back(idx);
            else rightIndices.push_back(idx);
        }

        return { leftIndices, rightIndices };
    }

    //Find best split for current node
    std::tuple<int, double, std::string, double> findBestSplit(const std::vector<Passenger>& data, const std::vector<int>& indices) {
        double bestGini = 1.0;
        int bestFeature = -1;
        double bestValue = 0.0;
        std::string bestCategory = "";

        std::random_device rd;
        std::mt19937 rng(rd());

        // Get current labels
        std::vector<bool> currentLabels;
        for (int idx : indices) {
            currentLabels.push_back(data[idx].survived);
        }
        double parentGini = calculateGini(currentLabels);

        if (featureSampleRatio > 1.0) featureSampleRatio = 1.0; //prevent failure incase a wrong value is passed.

        int nFeatures = std::max(1, (int)std::round(featureSampleRatio * 7));

        std::vector<int> featureIndices(7);
        std::iota(std::begin(featureIndices), std::end(featureIndices), 0);
        std::shuffle(std::begin(featureIndices), std::end(featureIndices), rng);

        std::vector<int> chosenFeatures(std::begin(featureIndices), std::begin(featureIndices) + nFeatures);


        // Try features
        for (int featureIdx : chosenFeatures) {
            // For numerical features
            if (featureIdx == 0 || featureIdx == 2 || featureIdx == 3 || featureIdx == 4 || featureIdx == 5) {
                // Try different split points
                std::vector<double> values;
                for (int idx : indices) {
                    double value = 0.0;
                    const Passenger& p = data[idx];

                    if (featureIdx == 0) value = p.pclass;
                    else if (featureIdx == 2) { if (p.age >= 0) value = p.age; else continue; }
                    else if (featureIdx == 3) value = p.sibSp;
                    else if (featureIdx == 4) value = p.parch;
                    else if (featureIdx == 5) { if (p.fare >= 0) value = p.fare; else continue; }

                    values.push_back(value);
                }

                if (values.empty()) continue;

                // Try unique values as potential splits
                std::sort(values.begin(), values.end());
                auto last = std::unique(values.begin(), values.end());
                values.erase(last, values.end());

                for (double value : values) {
                    auto [leftIndices, rightIndices] = splitData(data, indices, featureIdx, value);

                    if (leftIndices.size() < minSamplesLeaf || rightIndices.size() < minSamplesLeaf) {
                        continue;
                    }

                    // Calculate weighted Gini
                    std::vector<bool> leftLabels, rightLabels;
                    for (int idx : leftIndices) leftLabels.push_back(data[idx].survived);
                    for (int idx : rightIndices) rightLabels.push_back(data[idx].survived);

                    double leftGini = calculateGini(leftLabels);
                    double rightGini = calculateGini(rightLabels);

                    double weightedGini = (leftLabels.size() * leftGini + rightLabels.size() * rightGini) / indices.size();

                    if (weightedGini < bestGini) {
                        bestGini = weightedGini;
                        bestFeature = featureIdx;
                        bestValue = value;
                        bestCategory = "";
                    }
                }
            }
            // For categorical features (sex, embarked)
            else if (featureIdx == 1 || featureIdx == 6) {
                std::unordered_map<std::string, int> categoryCounts;
                for (int idx : indices) {
                    std::string category;
                    const Passenger& p = data[idx];

                    if (featureIdx == 1) category = p.sex;
                    else if (featureIdx == 6) category = p.embarked;

                    categoryCounts[category]++;
                }

                // Try each category as a split
                for (const auto& pair : categoryCounts) {
                    if (pair.second < minSamplesLeaf) continue;

                    auto [leftIndices, rightIndices] = splitData(data, indices, featureIdx, 0.0, pair.first);

                    if (leftIndices.size() < minSamplesLeaf || (indices.size() - leftIndices.size()) < minSamplesLeaf) {
                        continue;
                    }

                    // Calculate weighted Gini
                    std::vector<bool> leftLabels, rightLabels;
                    for (int idx : leftIndices) leftLabels.push_back(data[idx].survived);
                    for (int idx : rightIndices) rightLabels.push_back(data[idx].survived);

                    double leftGini = calculateGini(leftLabels);
                    double rightGini = calculateGini(rightLabels);

                    double weightedGini = (leftLabels.size() * leftGini + rightLabels.size() * rightGini) / indices.size();

                    if (weightedGini < bestGini) {
                        bestGini = weightedGini;
                        bestFeature = featureIdx;
                        bestValue = 0.0;
                        bestCategory = pair.first;
                    }
                }
            }
        }
        double gain = parentGini - bestGini;
        featureImportance[bestFeature] += gain;

        // Only return if there's actual gain
        if (bestGini < parentGini) {
            return { bestFeature, bestValue, bestCategory, bestGini };
        }

        return { -1, 0.0, "", 0.0 };
    }

    TreeNode* buildTree(const std::vector<Passenger>& data, const std::vector<int>& indices, int depth) {
        TreeNode* node = new TreeNode();
        //check stopping criteria
        if (depth >= maxDepth || indices.size() < minSamplesSplit) {
            node->isLeaf = true;
            //majority vote
            int count0 = 0, count1 = 0;
            for (int idx : indices) {
                if (data[idx].survived) ++count1;
                else ++count0;
            }
            node->leafClass = count1 > count0;
            return node;
        }

        //find best split
        auto [featureIdx, splitValue, splitCategory, gini] = findBestSplit(data, indices);
        if (featureIdx == -1) {
            node->isLeaf = true;
            //majority vote
            int count0 = 0, count1 = 0;
            for (int idx : indices) {
                if (data[idx].survived) ++count1;
                else ++count0;
            }
            node->leafClass = count1 > count0;
            return node;
        }

        //split the data
        auto [leftIndices, rightIndices] = splitData(data, indices, featureIdx, splitValue, splitCategory);

        if (leftIndices.size() < minSamplesLeaf || rightIndices.size() < minSamplesLeaf) {
            node->isLeaf = true;
            //majority vote
            int count0 = 0, count1 = 0;
            for (int idx : indices) {
                if (data[idx].survived) ++count1;
                else ++count0;
            }
            node->leafClass = count1 > count0;
            return node;
        }

        //create internal node
        node->isLeaf = false;
        node->featureIdx = featureIdx;
        node->splitValue = splitValue;
        node->splitCategory = splitCategory;
        node->left = buildTree(data, leftIndices, depth + 1);
        node->right = buildTree(data, rightIndices, depth + 1);
        return node;
    }

    int countLeaves(const TreeNode* node) const {
        if (!node) return 0;
        if (node->isLeaf) return 1;
        return countLeaves(node->left) + countLeaves(node->right);
    }

    int treeDepth(const TreeNode* node) const {
        if (!node) return 0;
        if (node->isLeaf) return 1;
        return 1 + std::max(treeDepth(node->left), treeDepth(node->right));
    }

    void copyTree(TreeNode*& toNode, TreeNode* fromNode) {
        if (!fromNode) toNode = nullptr;
        else {
            toNode = new TreeNode;
            toNode->featureIdx = fromNode->featureIdx;
            toNode->splitValue = fromNode->splitValue;
            toNode->splitCategory = fromNode->splitCategory;
            toNode->isLeaf = fromNode->isLeaf;
            toNode->leafClass = fromNode->leafClass;
            copyTree(toNode->left, fromNode->left);
            copyTree(toNode->right, fromNode->right);
        }
    }

public:
    DecisionTree(int maxDepth = 5, int minSamplesSplit = 2, int minSamplesLeaf = 1, double featureSampleRatio = 1.0) :
        root(nullptr), maxDepth(maxDepth), minSamplesSplit(minSamplesSplit), minSamplesLeaf(minSamplesLeaf), featureSampleRatio(featureSampleRatio) {}

    DecisionTree(const DecisionTree& other) {
        if (!other.root)
            root = nullptr;
        else
            copyTree(root, other.root);
        maxDepth = other.maxDepth;
        minSamplesSplit = other.minSamplesSplit;
        minSamplesLeaf = other.minSamplesLeaf;
        featureSampleRatio = other.featureSampleRatio;
        featureImportance = other.featureImportance;
    }

    DecisionTree(DecisionTree&& other) {
        std::swap(root, other.root);
        maxDepth = std::move(other.maxDepth);
        minSamplesSplit = std::move(other.minSamplesSplit);
        minSamplesLeaf = std::move(other.minSamplesLeaf);
        featureSampleRatio = std::move(other.featureSampleRatio);
        featureImportance = std::move(other.featureImportance);
    }

    DecisionTree operator=(const DecisionTree& other) {
        if (!other.root)
            root = nullptr;
        else
            copyTree(root, other.root);
        maxDepth = other.maxDepth;
        minSamplesSplit = other.minSamplesSplit;
        minSamplesLeaf = other.minSamplesLeaf;
        featureSampleRatio = other.featureSampleRatio;
        featureImportance = other.featureImportance;
        return *this;
    }

    DecisionTree operator=(DecisionTree&& other) {
        std::swap(root, other.root);
        maxDepth = std::move(other.maxDepth);
        minSamplesSplit = std::move(other.minSamplesSplit);
        minSamplesLeaf = std::move(other.minSamplesLeaf);
        featureSampleRatio = std::move(other.featureSampleRatio);
        featureImportance = std::move(other.featureImportance);
        return *this;
    }

    ~DecisionTree() {
        delete root;
    }

    void train(const std::vector<Passenger>& data) {
        std::vector<int> indices(data.size());
        std::iota(std::begin(indices), std::end(indices), 0);
        root = buildTree(data, indices, 0);
        /* std::cout << "Tree depth: " << treeDepth(root)
             << ", Leaves: " << countLeaves(root) << "\n";*/
    }

    bool predict(const Passenger& p) const {
        const TreeNode* node = root;
        while (node && !node->isLeaf) {
            bool goLeft = false;
            if (node->featureIdx == 0) // pclass
                goLeft = p.pclass <= node->splitValue;
            else if (node->featureIdx == 1) // sex
                goLeft = p.sex == node->splitCategory;
            else if (node->featureIdx == 2) // age
                goLeft = p.age <= node->splitValue && p.age >= 0;
            else if (node->featureIdx == 3) // sibSp
                goLeft = p.sibSp <= node->splitValue;
            else if (node->featureIdx == 4) // parch
                goLeft = p.parch <= node->splitValue;
            else if (node->featureIdx == 5) // fare
                goLeft = p.fare <= node->splitValue && p.fare >= 0;
            else if (node->featureIdx == 6) // embarked
                goLeft = p.embarked == node->splitCategory;
            node = goLeft ? node->left : node->right;
        }
        return node ? node->leafClass : false;
    }

    std::unordered_map<int, double> getFeatureImportance() const {
        return featureImportance;
    }
};