#include "Actor.h"

ActorImpl::ActorImpl() {
	this->L1 = torch::nn::Linear(4, 24);
	this->L2 = torch::nn::Linear(24, 24);
	this->L3 = torch::nn::Linear(24, 1);

	this->initializeParameters();

	this->net = this->register_module(
		"net",
		torch::nn::Sequential(
			torch::nn::BatchNorm1d(4),
			this->L1,
			torch::nn::ReLU(),
			torch::nn::Dropout(0.6),
			this->L2,
			torch::nn::ReLU(),
			torch::nn::Dropout(0.6),
			this->L3,
			torch::nn::Tanh()
		)
	);
}

void ActorImpl::copyHardWeight(ActorImpl source) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(source.net->parameters()[i]).detach_();
	}
}

void ActorImpl::copySoftWeigth(ActorImpl source, double tau) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(tau * source.net->parameters()[i] + (1 - tau) * this->net->parameters()[i]).detach_();
	}
}

torch::Tensor ActorImpl::forward(torch::Tensor input) {
	return this->net->forward(input);
}

void ActorImpl::initializeParameters()
{
	/*
	torch::nn::init::xavier_uniform_(this->L1->weight);
	torch::nn::init::xavier_uniform_(this->L2->weight);
	torch::nn::init::xavier_uniform_(this->L3->weight);
	torch::nn::init::xavier_uniform_(this->L4->weight);
	*/

	torch::nn::init::kaiming_uniform_(this->L1->weight, 0.0, torch::kFanIn, torch::kReLU);
	torch::nn::init::kaiming_uniform_(this->L2->weight, 0.0, torch::kFanIn, torch::kReLU);
	torch::nn::init::kaiming_uniform_(this->L3->weight, 0.0, torch::kFanIn, torch::kReLU);
	//torch::nn::init::kaiming_uniform_(this->L4->weight, 0.0, torch::kFanIn, torch::kReLU);

	torch::nn::init::zeros_(this->L1->bias);
	torch::nn::init::zeros_(this->L2->bias);
	torch::nn::init::zeros_(this->L3->bias);
	//torch::nn::init::zeros_(this->L4->bias);
}
