// This is a very simple implementation of a multilayer perceptron, in large
// parts translated from the pyton code at neuralnetworksanddeeplearning.com
//
// It is very ineffiecient but the idea here is to showcase the basic mathematical
// workings behind this simple neural network and familiarizing myself again
// with (modern) c++
//


#include <iostream>
#include <vector>

#include "mnist/mnist_reader.hpp"

#include "network.h"

using namespace std;

int main(){
    cout << "loading dataset..." << endl;
    auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>();
    
    cout << "finished loading " << dataset.training_images.size() << " training and ";
    cout << dataset.test_images.size() << " test images" << endl;

    Network net = Network(vector<int> {784, 30, 10});
    net.PrintLayers();

    net.Sgd(dataset.training_images,
            dataset.training_labels,
            30, 10, 3.0, 
            dataset.test_images,
            dataset.test_labels);
    

    return 0;
}
