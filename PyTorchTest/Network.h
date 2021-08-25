#pragma once
#include <torch/torch.h>

class NetworkImpl:public torch::nn::Module
{
public:
	NetworkImpl();

	torch::Tensor forward(torch::Tensor x);
private:
	torch::nn::Linear L1{ nullptr };
	torch::nn::Linear L2{ nullptr };
	torch::nn::Linear L3{ nullptr };

	torch::nn::Sequential net;
};

TORCH_MODULE(Network);