// BackPropagation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include"NeuralNet.h"

int main()
{
    float inputs[] = {0.5f,0.0f};

    NeuralNet nn = NeuralNet(2, 2, 1, 2);
    nn.randomizeWeights();
    nn.randomizeBiases();
    nn.setInputs(inputs);
    nn.forwardPropagate();

    return 0;
}
