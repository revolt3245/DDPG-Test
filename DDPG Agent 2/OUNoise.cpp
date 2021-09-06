#include "OUNoise.h"

OUNoise& OUNoise::setMu(torch::Tensor mu)
{
	this->mu = mu.detach();

	return *this;
}

void OUNoise::reset(torch::Tensor x0)
{
	this->x = x0;
}

torch::Tensor OUNoise::getNoise()
{
	auto dx = (this->mu - this->x) * dt + this->sigma * torch::randn_like(this->x);
	this->x += dx;
	return this->x;
}
