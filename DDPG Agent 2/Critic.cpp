#include "Critic.h"

CriticImpl::CriticImpl() {
	this->aL1 = torch::nn::Linear(1, 100);
	this->sL1 = torch::nn::Linear(4, 100);
	this->sL2 = torch::nn::Linear(100, 100);

	this->L1 = torch::nn::Linear(200, 100);
	this->L2 = torch::nn::Linear(100, 1);
	//this->L3 = torch::nn::Linear(400, 1);

	this->initializeParameters();

	this->net = this->register_module(
		"net", 
		torch::nn::Sequential(
			torch::nn::ReLU(),
			torch::nn::Dropout(0.6),
			this->L1,
			torch::nn::ReLU(),
			torch::nn::Dropout(0.6),
			this->L2
		)
	);

	this->aBranch = this->register_module(
		"aBranch",
		torch::nn::Sequential(
			this->aL1
		)
	);

	this->sBranch = this->register_module(
		"sBranch",
		torch::nn::Sequential(
			torch::nn::BatchNorm1d(4),
			this->sL1,
			torch::nn::ReLU(),
			torch::nn::Dropout(0.6),
			this->sL2
		)
	);
}

void CriticImpl::copyHardWeight(CriticImpl source) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(source.net->parameters()[i]).detach_();
	}
}
void CriticImpl::copySoftWeight(CriticImpl source, double tau) {
	torch::NoGradGuard noGrad;

	for (auto i = 0; i < this->net->parameters().size(); i++) {
		this->net->parameters()[i].copy_(tau * source.net->parameters()[i] + (1 - tau) * this->net->parameters()[i]).detach_();
	}
}

torch::nn::Sequential CriticImpl::getNetwork() {
	return this->net;
}

void CriticImpl::initializeParameters()
{
	torch::nn::init::kaiming_uniform_(this->aL1->weight);

	torch::nn::init::kaiming_uniform_(this->sL1->weight);
	torch::nn::init::kaiming_uniform_(this->sL2->weight);

	torch::nn::init::kaiming_uniform_(this->L1->weight);
	torch::nn::init::kaiming_uniform_(this->L2->weight);
	//torch::nn::init::kaiming_uniform_(this->L3->weight);

	torch::nn::init::zeros_(this->aL1->bias);

	torch::nn::init::zeros_(this->sL1->bias);
	torch::nn::init::zeros_(this->sL2->bias);

	torch::nn::init::zeros_(this->L1->bias);
	torch::nn::init::zeros_(this->L2->bias);
	//torch::nn::init::zeros_(this->L3->bias);
}

torch::Tensor CriticImpl::forward(const torch::Tensor& state, const torch::Tensor& action)
{
	auto aB = this->aBranch->forward(action);
	auto sB = this->sBranch->forward(state);

	return this->net->forward(torch::cat({aB, sB}, 1));
}

