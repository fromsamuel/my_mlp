#include <iostream>
#include <vector>

#include "mnist/mnist_reader.hpp"

#include "network.h"

int main(){


	std::cout << "loading dataset..." << std::endl;

	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	
	std::cout << "finished loading " << dataset.training_images.size() << " training and ";
	std::cout << dataset.test_images.size() << " test images" << std::endl;

	std::cout << dataset.training_images[0].size() << std::endl;
	std::cout << +dataset.training_labels[0] << std::endl;

	//Network net = Network(std::vector<int> {784, 30, 10});
	Network net = Network(std::vector<int> {784, 30, 10});
	net.PrintLayers();

	net.Sgd(dataset.training_images,
		       dataset.training_labels,
	       	       30, 10, 3.0, dataset.test_images,
		       dataset.test_labels);
	

	return 0;
}
