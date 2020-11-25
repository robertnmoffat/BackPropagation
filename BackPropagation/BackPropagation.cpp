// BackPropagation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include"NeuralNet.h"

int main()
{
    float inputs[] = {0.1f,0.05f};

    float inputNeurons[] = {0.1f, 0.05f};
    float inputWeights[] = {0.3f, 0.25f, 0.2f, 0.15f};
    float outputWeights[] = {0.55f, 0.5f, 0.45f, 0.4f};
    //float hiddenWeights[0][0];

    float** biases;
    biases = new float* [2];
    for (int i = 0; i < 2; i++) {
        biases[i] = new float[2];
        for (int j = 0; j < 2; j++) {
            biases[i][j] = 0.0f;
        }
    }
    biases[0][0] = 0.35f;
    biases[0][1] = 0.35f;
    biases[1][0] = 0.6f;
    biases[1][1] = 0.6f;

    NeuralNet nn = NeuralNet(2, 2, 1, 2);
    nn.randomizeWeights();
    nn.randomizeBiases();
    //nn.setValues(inputs, inputWeights, outputWeights, NULL, biases);
    nn.setInputs(inputs);
    nn.forwardPropagate();

    return 0;
}
