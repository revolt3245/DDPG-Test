#include "Network.h"

NetworkImpl::NetworkImpl()
{
	this->L1 = torch::nn::Linear(2, 10);
	this->L2 = torch::nn::Linear(10, 10);
	this->L3 = torch::nn::Linear(10, 1);



	this->net = this->register_module(
		"net",
		torch::nn::Sequential(
			this->L1,
			torch::nn::ReLU(),
			this->L2,
			torch::nn::ReLU(),
			this->L3,
			torch::nn::Sigmoid()
		)
	);
}

torch::Tensor NetworkImpl::forward(torch::Tensor x)
{
	return this->net->forward(x);
}
