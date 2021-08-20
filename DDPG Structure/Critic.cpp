#include "Critic.h"

Critic::Critic() {
	this->net = torch::nn::Sequential(
		torch::nn::Linear(8,100),
		torch::nn::ReLU(),
		torch::nn::Linear(100, 100),
		torch::nn::ReLU(),
		torch::nn::Linear(100, 100),
		torch::nn::ReLU(),
		torch::nn::Linear(100, 1)
	);
}

torch::Tensor Critic::forward(torch::Tensor input) {
	return this->net->forward(input);
}