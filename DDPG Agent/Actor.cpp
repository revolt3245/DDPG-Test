#include "Actor.h"

Actor::Actor() {
	this->L1 = torch::nn::Linear(6, 5);
	this->L2 = torch::nn::Linear(5, 5);
	this->L3 = torch::nn::Linear(5, 5);
	this->L4 = torch::nn::Linear(5, 2);

	this->net = torch::nn::Sequential(
		this->L1,
		torch::nn::ReLU(),
		this->L2,
		torch::nn::ReLU(),
		this->L3,
		torch::nn::ReLU(),
		this->L4,
		torch::nn::Tanh()
	);
}

void Actor::copyHardWeight(Actor source) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(source.net->parameters()[i]);
	}
}

void Actor::copySoftWeigth(Actor source, double tau) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(tau * source.net->parameters()[i] + (1 - tau) * this->net->parameters()[i]);
	}
}

torch::nn::Sequential Actor::getNetwork() {
	return this->net;
}

torch::Tensor Actor::forward(torch::Tensor input) {
	return this->net->forward(input);
}