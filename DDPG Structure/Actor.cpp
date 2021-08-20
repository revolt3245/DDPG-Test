#include "Actor.h"

Actor::Actor() {
	this->net = torch::nn::Sequential(
		torch::nn::Linear(6, 100),
		torch::nn::ReLU(),
		torch::nn::Linear(100, 100),
		torch::nn::ReLU(),
		torch::nn::Linear(100, 100),
		torch::nn::ReLU(),
		torch::nn::Linear(100, 2),
		torch::nn::Tanh()
	);
}

torch::Tensor Actor::forward(torch::Tensor input) {
	return this->net->forward(input);
}