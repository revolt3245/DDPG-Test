#pragma once

#include <torch/torch.h>
#include <iostream>

using namespace std;

class Actor :public torch::nn::Module
{
public:
	Actor();

	torch::Tensor forward(torch::Tensor input);

	torch::nn::Sequential getNetwork();

	void copyHardWeight(Actor source);
	void copySoftWeigth(Actor source, double tau);
private:
	torch::nn::Linear L1{ nullptr };
	torch::nn::Linear L2{ nullptr };
	torch::nn::Linear L3{ nullptr };
	torch::nn::Linear L4{ nullptr };

	torch::nn::Sequential net;
};

