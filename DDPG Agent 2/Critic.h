#pragma once

#include <torch/torch.h>

class CriticImpl :public torch::nn::Module
{
public:
	CriticImpl();

	torch::Tensor forward(const torch::Tensor& state, const torch::Tensor& action);

	torch::nn::Sequential getNetwork();

	void initializeParameters();

	void copyHardWeight(CriticImpl source);
	void copySoftWeight(CriticImpl source, double tau);
private:
	torch::nn::Linear aL1{ nullptr };

	torch::nn::Linear sL1{ nullptr };
	torch::nn::Linear sL2{ nullptr };

	torch::nn::Linear L1{ nullptr };
	torch::nn::Linear L2{ nullptr };
	//torch::nn::Linear L3{ nullptr };

	torch::nn::Sequential net;
	torch::nn::Sequential aBranch;
	torch::nn::Sequential sBranch;
};

TORCH_MODULE(Critic);