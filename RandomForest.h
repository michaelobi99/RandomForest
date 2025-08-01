#pragma once

#include "DecisionTree.h"

class RandomForest {
private:
	std::vector<DecisionTree> trees;
	int nTrees;
	int maxDepth;
	int minSamplesSplit;
	int minSamplesLeaf;
	double featureSampleRatio;

	//create bootstrap sample
	std::vector<int> createBootstrapSample(unsigned size, std::mt19937& rng) {
		std::vector<int> sample(size);
		std::uniform_int_distribution<int> dist(0, size - 1);
		for (int i = 0; i < size; ++i) {
			sample[i] = dist(rng);
		}
		return sample;
	}

public:
	RandomForest(int nTrees = 100, int maxDepth = 5, int minSamplesSplit = 2, int minSamplesLeaf = 1, double featureSampleRatio = 1.0) :
		nTrees(nTrees), maxDepth(maxDepth), minSamplesSplit(minSamplesSplit), minSamplesLeaf(minSamplesLeaf), featureSampleRatio(featureSampleRatio) {}

	
	void train(const std::vector<Passenger>& data) {
		std::random_device rd;
		std::mt19937 rng(rd());
		trees.reserve(nTrees);
		for (int i = 0; i < nTrees; ++i) {
			//create bootstrap sample
			auto sampleIndices = createBootstrapSample(data.size(), rng);
			//create and train tree
			DecisionTree tree(maxDepth, minSamplesSplit, minSamplesLeaf, featureSampleRatio);

			//create sampled dataset
			std::vector<Passenger> sampleData;
			for (int idx : sampleIndices)
				sampleData.push_back(data[idx]);

			tree.train(sampleData);
			trees.push_back(tree);
		}
	}

	bool predict(const Passenger& p) const {
		int votes0 = 0, votes1 = 0;
		for (const DecisionTree& tree : trees) {
			int pred = tree.predict(p);
			if (pred == 0) votes0++;
			else votes1++;
		}
		return votes1 > votes0 ? 1 : 0;
	}

	double evaluate(const std::vector<Passenger>& testData) const {
		int correct = 0;
		for (const auto& p : testData) {
			if (predict(p) == p.survived) ++correct;
		}
		return static_cast<double>(correct) / testData.size();
	}

	std::unordered_map<int, double> computeFeatureImportances() {
		std::unordered_map<int, double> total;
		for (const auto& tree : trees) {
			auto imp = tree.getFeatureImportance();
			for (auto& [feature, score] : imp) {
				total[feature] += score;
			}
		}
		double sum = 0;
		for (auto& [_, score] : total) sum += score;
		if (sum > 0) {
			for (auto& [_, score] : total) score /= sum;
		}
		return total;
	}

	void save(const std::string& model_file) {
		std::fstream file(model_file, std::ios::out | std::ios::binary);
		file.write(reinterpret_cast<const char*>(&nTrees), sizeof(nTrees));
		for (DecisionTree& tree : trees) {
			tree.save(file);
		}
		file.close();
	}

	void load(const std::string& model_file) {
		std::fstream file(model_file, std::ios::in | std::ios::binary);
		file.read(reinterpret_cast<char*>(&nTrees), sizeof(nTrees));
		if (nTrees > 0) {
			trees.reserve(nTrees);
			for (int i{ 0 }; i < nTrees; ++i) {
				DecisionTree tree;
				tree.load(file);
				trees.push_back(tree);
			}
		}
		file.close();
	}
};