#pragma once

#include <torch/torch.h>

class Critic :public torch::nn::Module
{
public:
	Critic();

	torch::Tensor forward(torch::Tensor input);

	torch::nn::Sequential getNetwork();

	void copyHardWeight(Critic source);
	void copySoftWeight(Critic source, double tau);
private:
	torch::nn::Linear L1{ nullptr };
	torch::nn::Linear L2{ nullptr };
	torch::nn::Linear L3{ nullptr };
	torch::nn::Linear L4{ nullptr };

	torch::nn::Sequential net;
};

