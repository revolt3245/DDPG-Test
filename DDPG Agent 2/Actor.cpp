#include "Actor.h"

ActorImpl::ActorImpl() {
	this->L1 = torch::nn::Linear(4, 100);
	this->L2 = torch::nn::Linear(100, 100);
	this->L3 = torch::nn::Linear(100, 100);
	this->L4 = torch::nn::Linear(100, 1);

	this->net = this->register_module(
		"net", 
		torch::nn::Sequential(
			this->L1,
			torch::nn::ReLU(),
			this->L2,
			torch::nn::ReLU(),
			this->L3,
			torch::nn::ReLU(),
			this->L4,
			torch::nn::Tanh()
		)
	);
}

void ActorImpl::copyHardWeight(ActorImpl source) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(source.net->parameters()[i]);
	}
}

void ActorImpl::copySoftWeigth(ActorImpl source, double tau) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(tau * source.net->parameters()[i] + (1 - tau) * this->net->parameters()[i]);
	}
}

torch::Tensor ActorImpl::forward(torch::Tensor input) {
	return this->net->forward(input);
}
