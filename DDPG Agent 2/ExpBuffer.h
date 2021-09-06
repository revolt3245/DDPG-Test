#pragma once
#include <torch/torch.h>
#include <array>
#include <iostream>

using namespace std;

struct Exp {
	torch::Tensor current;
	torch::Tensor action;
	torch::Tensor reward;
	torch::Tensor next;
	torch::Tensor terminate;

	Exp() :current(), action(), reward(), next(), terminate() {
	};
	Exp(torch::Tensor current, torch::Tensor action, torch::Tensor reward, torch::Tensor next, torch::Tensor terminate) :current(current), action(action), reward(reward), next(next), terminate(terminate) {
	};

	friend ostream& operator<<(ostream& os, Exp exp);

	Exp& operator=(const Exp& obj);

	Exp cpu();
	Exp cuda();

	Exp to(torch::Device device);
};

class ExpBuffer
{
public:
	ExpBuffer() :Buffer(0), Size(1e6) {};
	ExpBuffer(size_t Size) :Buffer(0), Size(Size) {};

	~ExpBuffer() {};

	size_t getMaxSize();
	void setMaxSize(size_t Size);

	size_t getSize();

	void push(Exp experience);
	bool isEmpty();

	Exp sample();
	Exp sample(unsigned int idx);
	Exp sample(torch::Tensor indices);
	Exp sampleBatch(size_t sampleSize);
	Exp sampleBatch(size_t sampleSize, torch::Device device);
private:
	size_t Size;
	std::vector<Exp> Buffer;
};

