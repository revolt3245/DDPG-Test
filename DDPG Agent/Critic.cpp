#include "Critic.h"

Critic::Critic() {
	this->L1 = torch::nn::Linear(8, 100);
	this->L2 = torch::nn::Linear(100, 100);
	this->L3 = torch::nn::Linear(100, 100);
	this->L4 = torch::nn::Linear(100, 1);

	this->net = torch::nn::Sequential(
		this->L1,
		torch::nn::ReLU(),
		this->L2,
		torch::nn::ReLU(),
		this->L3,
		torch::nn::ReLU(),
		this->L4
	);
}

void Critic::copyHardWeight(Critic source) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(source.net->parameters()[i]);
	}
}
void Critic::copySoftWeight(Critic source, double tau) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(tau * source.net->parameters()[i] + (1 - tau) * this->net->parameters()[i]);
	}
}

torch::nn::Sequential Critic::getNetwork() {
	return this->net;
}

torch::Tensor Critic::forward(torch::Tensor input) {
	return this->net->forward(input);
}