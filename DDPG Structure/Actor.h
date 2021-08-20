#pragma once

#include <torch/torch.h>

class Actor:public torch::nn::Module
{
public:
	Actor();

	torch::Tensor forward(torch::Tensor input);
private:
	torch::nn::Sequential net;
};

