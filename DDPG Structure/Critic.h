#pragma once

#include <torch/torch.h>

class Critic:public torch::nn::Module
{
public:
	Critic();

	torch::Tensor forward(torch::Tensor input);
private:
	torch::nn::Sequential net;
};

