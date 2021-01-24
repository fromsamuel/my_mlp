#include "network.h"

#include "utils.h"

#include "Eigen/Dense"

#include <iostream>
#include <numeric>
#include <random>

Network::Network(std::vector<int> layers){
    this->layers_ = layers;
    
    for (uint32_t i=1; i<layers_.size(); i++){
        biases_.push_back(Eigen::MatrixXd::Random(layers_[i], 1));
    }

    for (uint32_t i=0; i<layers_.size()-1; i++){
        weights_.push_back(Eigen::MatrixXd::Random(layers_[i+1], layers_[i]));
    }
}

void Network::Backprop(std::vector<double> x,
                       uint8_t y,
                       std::vector<Eigen::MatrixXd> &delta_nabla_b,
                       std::vector<Eigen::MatrixXd> &delta_nabla_w){
    // feedforward
    std::vector<Eigen::VectorXd> activations, zs;
    Eigen::Map<Eigen::VectorXd> input = Eigen::Map<Eigen::VectorXd>(&x[0], x.size());
    input = input / 255;
    activations.push_back(input);
    
    Eigen::VectorXd z;
    for (uint8_t i=0; i<layers_.size()-1; ++i){
        Eigen::VectorXd z = weights_[i] * activations.back() + biases_[i];
        zs.push_back(z);
        activations.push_back(Sigmoid(z));
    }

    // backward pass
    Eigen::VectorXd delta = CostDerivative(activations.back(), y).array() * SigmoidPrime(zs.back()).array();
    delta_nabla_b.back() = delta;
    delta_nabla_w.back() = delta * activations.rbegin()[1].transpose();

    for (uint8_t i=1; i<layers_.size()-1; ++i){
        z = zs.rbegin()[i];
        Eigen::VectorXd sp = SigmoidPrime(z);
        delta = (weights_.rbegin()[i-1].transpose() * delta).array() * sp.array();
        delta_nabla_b.rbegin()[i] = delta;
        delta_nabla_w.rbegin()[i] = delta * activations.rbegin()[i+1].transpose();
    }
}

void Network::UpdateMiniBatch(std::vector<std::vector<uint8_t>> mini_batch,
                              std::vector<uint8_t> labels,
                              double eta){
    std::vector<Eigen::MatrixXd> nabla_b;
    std::vector<Eigen::MatrixXd> nabla_w;
    
    for (uint32_t i=1; i<layers_.size(); ++i){
        nabla_b.push_back(Eigen::MatrixXd::Zero(layers_[i], 1));
    }
    for (uint32_t i=0; i<layers_.size()-1; ++i){
        nabla_w.push_back(Eigen::MatrixXd::Zero(layers_[i+1], layers_[i]));
    }
    
    //backprop sample by sample, could be made more efficient
    for (uint32_t i=0; i<labels.size(); ++i){
        std::vector<Eigen::MatrixXd> delta_nabla_b, delta_nabla_w;
        
        for (uint32_t i=1; i<layers_.size(); ++i){
            delta_nabla_b.push_back(Eigen::MatrixXd::Zero(layers_[i], 1));
        }
        for (uint32_t i=0; i<layers_.size()-1; ++i){
            delta_nabla_w.push_back(Eigen::MatrixXd::Zero(layers_[i+1], layers_[i]));
        }

        std::vector<double> x(mini_batch[i].begin(), mini_batch[i].end());      
        Backprop(x, labels[i], delta_nabla_b, delta_nabla_w);

        for (uint8_t i=0; i<layers_.size()-1; ++i){
            nabla_w[i] = nabla_w[i] + delta_nabla_w[i];
            nabla_b[i] = nabla_b[i] + delta_nabla_b[i];
        }
    }

    // update weights and biases    
    for (uint8_t i=0; i<layers_.size()-1; ++i){
        weights_[i] = weights_[i] - (eta/mini_batch.size())*nabla_w[i];
        biases_[i] = biases_[i] - (eta/mini_batch.size())*nabla_b[i];
    }
}

void Network::PrintLayers(int verbosity){
    std::cout << "Multilayer Perceptron Structure: ";
    for (int layer : this->layers_){
        std::cout << layer << " ";
    }
    std::cout << std::endl;

    if (verbosity > 1){
        for (auto biases : this->biases_){
            std::cout << biases << std::endl;
        }

        for (auto weights : this->weights_){
            std::cout << weights.rows() << ":" << weights.cols() << std::endl;
        }
    }
}

int Network::Evaluate(std::vector<std::vector<unsigned char>> data,
                      std::vector<unsigned char> labels){

    Eigen::VectorXd result;
    Eigen::Index maxInd;
    int total = 0;
    for(uint32_t i = 0; i < data.size(); ++i){
        std::vector<double> doubleVec(data[i].begin(), data[i].end());  
        result = FeedForward(doubleVec);
        result.maxCoeff(&maxInd);
        if (maxInd == labels[i]){
            total += 1;
        }
    }
    return total;
}

void Network::Sgd(std::vector<std::vector<uint8_t>> training_data,
                  std::vector<uint8_t> training_labels,
                  int epochs,
                  int mini_batch_size, 
                  double eta,
                  std::vector<std::vector<uint8_t>> test_data,
                  std::vector<uint8_t> test_labels){
    
    int n = training_data.size();
    int n_test = test_data.size();

    for (int i = 0; i < epochs; i++){
        // add shuffling
	//
        for (int j = 0; j < n / mini_batch_size; j++){
            auto mini_data = std::vector<std::vector<uint8_t>>(training_data.begin()+mini_batch_size*j,
                    training_data.begin()+mini_batch_size*(j+1));
            
            auto mini_labels = std::vector<uint8_t>(training_labels.begin()+mini_batch_size*j,
                    training_labels.begin() + mini_batch_size*(j+1));
            
            UpdateMiniBatch(mini_data, mini_labels, eta);   
        }

        int num_correct = Evaluate(test_data, test_labels);
        std::cout << "Epoch " << i << " " <<  num_correct << "/" <<
            n_test << std::endl;
    }
}


Eigen::VectorXd Network::FeedForward(std::vector<double> input){
    Eigen::VectorXd result;
    double *ptr = &input[0];
    Eigen::Map<Eigen::VectorXd> input_vec = Eigen::Map<Eigen::VectorXd>(ptr, input.size());
    result = input_vec/255;
    for (uint8_t i = 0; i < layers_.size()-1; ++i){
        result = Sigmoid(weights_[i] * result + biases_[i]);
    }
    return result;
}

Eigen::MatrixXd Network::Sigmoid(Eigen::MatrixXd z){
    return 1.0 / (1.0 + Eigen::exp(-z.array()));
}

Eigen::MatrixXd Network::SigmoidPrime(Eigen::MatrixXd z){
    return Sigmoid(z).array() *  (Eigen::MatrixXd::Ones(z.rows(), z.cols()) - Sigmoid(z)).array();

}

Eigen::VectorXd Network::CostDerivative(Eigen::VectorXd output_activations, uint8_t y){
    Eigen::VectorXd y_oh = Eigen::VectorXd::Zero(output_activations.rows());
    y_oh[y] = 1;
    return output_activations - y_oh;
}
