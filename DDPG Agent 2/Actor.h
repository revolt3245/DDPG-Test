#pragma once

#include <torch/torch.h>
#include <iostream>

using namespace std;

class ActorImpl :public torch::nn::Module
{
public:
	ActorImpl();

	torch::Tensor forward(torch::Tensor input);

	void initializeParameters();

	void copyHardWeight(ActorImpl source);
	void copySoftWeigth(ActorImpl source, double tau);
private:
	torch::nn::Linear L1{ nullptr };
	torch::nn::Linear L2{ nullptr };
	torch::nn::Linear L3{ nullptr };
	//torch::nn::Linear L4{ nullptr };

	torch::nn::Sequential net;
};

TORCH_MODULE(Actor);