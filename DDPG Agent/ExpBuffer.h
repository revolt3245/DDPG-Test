#pragma once
#include <torch/torch.h>
#include <array>

struct Exp {
	torch::Tensor current;
	torch::Tensor action;
	torch::Tensor reward;
	torch::Tensor next;

	Exp() :current(), action(), reward(), next() {};
	Exp(torch::Tensor current, torch::Tensor action, torch::Tensor reward, torch::Tensor next) :current(current), action(action), reward(reward), next(next) {};
};

class ExpBuffer
{
public:
	ExpBuffer():Buffer(0), Size(1e6) {};
	ExpBuffer(size_t Size) :Buffer(0), Size(Size) {};

	~ExpBuffer() {};

	size_t getMaxSize();
	void setMaxSize(size_t Size);

	size_t getSize();

	void push(Exp experience);
	bool isEmpty();

	Exp sample();
	std::vector<Exp> sampleBatch(size_t sampleSize);
private:
	size_t Size;
	std::vector<Exp> Buffer;
};

