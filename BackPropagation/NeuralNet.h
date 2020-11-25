#pragma once

//#include<stdlib.h>
#include<cstdlib>
#include<iostream>
#include <time.h>

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

	float** biases;

	float weightStartRange = 2.0f;

public:
	NeuralNet(int inputNeuronCount, int outputNeuronCount, int hiddenLayerCount, int neuronsPerLayerCount){
		std::clog << "Initilizing network..." << std::endl;
		srand(time(0));

		this->inputNeuronCount = inputNeuronCount;
		this->outputNeuronCount = outputNeuronCount;
		this->hiddenLayerCount = hiddenLayerCount;
		this->neuronsPerLayerCount = neuronsPerLayerCount;

		inputNeurons = new float[inputNeuronCount];
		outputNeurons = new float[outputNeuronCount];
		hiddenNeurons = new float* [hiddenLayerCount];
		for (int i = 0; i < hiddenLayerCount; i++)
			hiddenNeurons[i] = new float[neuronsPerLayerCount];
		
		inputWeights = new float[inputNeuronCount*neuronsPerLayerCount];
		hiddenWeights = new float* [hiddenLayerCount-1];
		for (int i = 0; i < hiddenLayerCount - 1; i++)
			hiddenWeights[i] = new float[neuronsPerLayerCount*neuronsPerLayerCount];
		outputWeights = new float[neuronsPerLayerCount*outputNeuronCount];

		biases = new float* [hiddenLayerCount];
		for (int i = 0; i < hiddenLayerCount; i++)
			biases[i] = new float[neuronsPerLayerCount];
	}

	~NeuralNet() {
		std::clog << "Freeing memory..." << std::endl;
		free(inputNeurons);
		free(outputNeurons);
		free(hiddenNeurons);
		free(inputWeights);
		free(hiddenWeights);
		free(outputWeights);
	}

	void randomizeWeights() {
		std::clog << "Randomizing weights..." << std::endl;
		std::clog << "Input weights:" << std::endl;
		for (int i = 0; i < (inputNeuronCount * neuronsPerLayerCount); i++) {
			inputWeights[i] = randomWeight();
			std::clog << inputWeights[i] << std::endl;
		}

		std::clog << "Hidden weights:" << std::endl;
		for (int i = 0; i < hiddenLayerCount - 1; i++) {
			for (int j = 0; j < neuronsPerLayerCount * neuronsPerLayerCount; j++) {
				hiddenWeights[i][j] = randomWeight();
				std::clog << hiddenWeights[i][j] << std::endl;
			}
		}

		std::clog << "Output weights:" << std::endl;
		for (int i = 0; i < (neuronsPerLayerCount * outputNeuronCount); i++) {
			outputWeights[i] = randomWeight();
			std::clog << outputWeights[i] << std::endl;
		}
	}

	float randomWeight() {
		//return (rand() % (int)weightStartRange) - (weightStartRange / 2);
		return (float)(rand() % 200) / 100.0f - 1.0f;
	}

	void randomizeBiases() {
		std::clog << "Randomizing biases..." << std::endl;
		for (int i = 0; i < hiddenLayerCount; i++) {
			for (int j = 0; j < neuronsPerLayerCount; j++) {
				biases[i][j] = 0.0f;
			}
		}
	}

	void setInputs(float* inputs) {
		std::clog << "Setting inputs..." << std::endl;
		for (int i = 0; i < inputNeuronCount; i++) {
			inputNeurons[i] = inputs[i];
			std::clog << inputNeurons[i] << std::endl;
		}
	}

	float sigmoid(float x)
	{
		float exp_value;
		float return_value;

		/*** Exponential calculation ***/
		exp_value = exp((double)-x);

		/*** Final sigmoid value ***/
		return_value = 1 / (1 + exp_value);

		return return_value;
	}

	void forwardPropagate() {
		std::clog << "Forward propagation..." << std::endl;
		//--------Input to first hidden layer-----------
		for (int i = 0; i < neuronsPerLayerCount; i++) {
			hiddenNeurons[0][i] = 0.0f;
			for (int j = 0; j < inputNeuronCount; j++) {
				hiddenNeurons[0][i] += inputNeurons[j] * inputWeights[i*inputNeuronCount+j];
				//std::clog << hiddenNeurons[0][i] << std::endl;
			}
			hiddenNeurons[0][i] += biases[0][i];
			hiddenNeurons[0][i] = sigmoid(hiddenNeurons[0][i]);
			std::clog << "Neuron: " << "0," << i << " "<< hiddenNeurons[0][i] << std::endl;
		}

		//------------hidden layers---------------
		for (int i = 1; i < hiddenLayerCount; i++) {//start at 1 because the first layer was done above
			for (int j = 0; j < neuronsPerLayerCount; j++) {
				hiddenNeurons[i][j] = 0.0f;
				for (int k = 0; k < neuronsPerLayerCount; k++) {
					hiddenNeurons[i][j] += hiddenNeurons[i - 1][k] * hiddenWeights[i-1][j*neuronsPerLayerCount+k];
				}
				hiddenNeurons[i][j] += biases[i][j];
				hiddenNeurons[i][j] = sigmoid(hiddenNeurons[i][j]);
				std::clog << "Neuron: " << i << "," << j << " " << hiddenNeurons[i][j] << std::endl;
			}
		}

		//-------last hidden layer to output----------
		for (int i = 0; i < outputNeuronCount; i++) {
			outputNeurons[i] = 0.0f;
			for (int j = 0; j < neuronsPerLayerCount; j++) {
				outputNeurons[i] += hiddenNeurons[hiddenLayerCount-1][j] * outputWeights[i * neuronsPerLayerCount + j];
			}
			outputNeurons[i] = sigmoid(outputNeurons[i]);
			std::clog << "Output neuron: " << i << " " << outputNeurons[i] << std::endl;
		}
	}

};