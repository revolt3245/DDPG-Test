#include "Critic.h"

CriticImpl::CriticImpl() {
	this->aL1 = torch::nn::Linear(2, 100);
	this->sL1 = torch::nn::Linear(6, 100);
	this->L2 = torch::nn::Linear(100, 100);
	this->L3 = torch::nn::Linear(100, 100);
	this->L4 = torch::nn::Linear(100, 1);

	this->net = this->register_module(
		"net", 
		torch::nn::Sequential(
			this->L2,
			torch::nn::ReLU(),
			this->L3,
			torch::nn::ReLU(),
			this->L4
		)
	);

	this->aBranch = this->register_module(
		"aBranch",
		torch::nn::Sequential(
			this->aL1,
			torch::nn::ReLU()
		)
	);

	this->sBranch = this->register_module(
		"sBranch",
		torch::nn::Sequential(
			this->sL1,
			torch::nn::ReLU()
		)
	);
}

void CriticImpl::copyHardWeight(CriticImpl source) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(source.net->parameters()[i]);
	}
}
void CriticImpl::copySoftWeight(CriticImpl source, double tau) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(tau * source.net->parameters()[i] + (1 - tau) * this->net->parameters()[i]);
	}
}

torch::nn::Sequential CriticImpl::getNetwork() {
	return this->net;
}

torch::Tensor CriticImpl::forward(const torch::Tensor& state, const torch::Tensor& action)
{
	auto aB = this->aBranch->forward(action);
	auto sB = this->sBranch->forward(state);

	return this->net->forward(aB + sB);
}

