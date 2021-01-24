#include <vector>
#include "Eigen/Dense"


class Network{
    private: 
        std::vector<int> layers_;
        std::vector<Eigen::MatrixXd> biases_;
        std::vector<Eigen::MatrixXd> weights_;

        Eigen::VectorXd FeedForward(std::vector<double> input);
        Eigen::MatrixXd Sigmoid(Eigen::MatrixXd z);
        Eigen::MatrixXd SigmoidPrime(Eigen::MatrixXd z);
        Eigen::VectorXd CostDerivative(Eigen::VectorXd output_activations, uint8_t y);
        void UpdateMiniBatch(std::vector<std::vector<uint8_t>> mini_batch, 
                             std::vector<uint8_t> labels,
                             double eta);
        void Backprop(std::vector<double> x,
                      uint8_t y,
                      std::vector<Eigen::MatrixXd> &delta_nabla_b,
                      std::vector<Eigen::MatrixXd> &delta_nalba_w);
        int Evaluate(std::vector<std::vector<unsigned char>> data,
                     std::vector<unsigned char> labels);


    public:
        Network(std::vector<int> layers);

        void PrintLayers(int verbosity=0);
        void Sgd(std::vector<std::vector<uint8_t>> training_data,
                std::vector<uint8_t> training_labels,
                int epochs,
                int mini_batch_size, double eta,
                std::vector<std::vector<uint8_t>> test_data,
                std::vector<uint8_t> test_labels);
        
};

