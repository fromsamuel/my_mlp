#include "utils.h"

#include "Eigen/Dense"

#include <vector>
#include <iostream>


void Utils::PrintDims(Eigen::MatrixXd in){
    std::cout << "dims: " << in.rows() << "," << in.cols() << std::endl;
}

void Utils::PrintVec(Eigen::VectorXd vec){                                              
    for(auto d: std::vector<double>(vec.data(), vec.data()+vec.size())){        
        std::cout << d << " ";                                           
    }                                                                        
    std::cout << std::endl;                                                  
}