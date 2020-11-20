#pragma once
#include <xmemory>


enum activationFunction {
	sigmoid
};

class NeuralNet {
	int hiddenLayerCount;
	int neuronsPerLayerCount;
	int outputNeuronCount;
	int inputNeuronCount;

	float* inputNeurons;
	float* outputNeurons;
	float** hiddenNeurons;
	float* inputWeights;
	float* outputWeights;
	float** hiddenWeights;

	NeuralNet(int inputNeuronCount, int outputNeuronCount, int hiddenLayerCount, int neuronsPerLayerCount){
		this->inputNeuronCount = inputNeuronCount;
		this->outputNeuronCount = outputNeuronCount;
		this->hiddenLayerCount = hiddenLayerCount;
		this->neuronsPerLayerCount = neuronsPerLayerCount;

		hiddenNeurons = new float* [hiddenLayerCount];
		for (int i = 0; i < hiddenLayerCount; i++)
			hiddenNeurons[i] = new float[neuronsPerLayerCount];

		inputNeurons = new float[inputNeuronCount];

	}

	~NeuralNet() {
		free(inputNeurons);
		

	}
};