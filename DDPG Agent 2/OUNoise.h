#pragma once

#include <torch/torch.h>

class OUNoise
{
public:
	OUNoise() :x(), mu(), sigma(0.3), dt(0.02) {};
	OUNoise(torch::Tensor x0) :x(x0.detach()), mu(torch::zeros_like(x0)),sigma(0.3), dt(0.02) {};
	OUNoise(torch::Tensor x0, double sigma, double dt) :x(x0.detach()), mu(zeros_like(x0)), sigma(sigma), dt(dt) {};

	OUNoise& setMu(torch::Tensor mu);

	void reset(torch::Tensor x0);

	torch::Tensor getNoise();
private:
	double sigma;
	double dt;

	torch::Tensor x;
	torch::Tensor mu;
};

