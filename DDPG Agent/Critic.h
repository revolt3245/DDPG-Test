#pragma once

#include <torch/torch.h>

class CriticImpl :public torch::nn::Module
{
public:
	CriticImpl();

	torch::Tensor forward(torch::Tensor input);

	torch::nn::Sequential getNetwork();

	void copyHardWeight(CriticImpl source);
	void copySoftWeight(CriticImpl source, double tau);
private:
	torch::nn::Linear L1{ nullptr };
	torch::nn::Linear L2{ nullptr };
	torch::nn::Linear L3{ nullptr };
	torch::nn::Linear L4{ nullptr };

	torch::nn::Sequential net;
};

TORCH_MODULE(Critic);