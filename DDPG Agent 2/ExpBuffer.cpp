#include "ExpBuffer.h"

size_t ExpBuffer::getMaxSize()
{
	return this->Size;
}

void ExpBuffer::setMaxSize(size_t Size)
{
	this->Size = Size;
}

size_t ExpBuffer::getSize()
{
	return this->Buffer.size();
}

void ExpBuffer::push(Exp experience)
{
	if (this->Buffer.size() >= this->Size) {
		Buffer.erase(this->Buffer.begin());
	}
	Buffer.push_back(experience);
}

bool ExpBuffer::isEmpty()
{
	return this->Buffer.empty();
}

Exp ExpBuffer::sample()
{
	if (this->Buffer.size() > 0) {
		srand(time(0));

		auto idx = rand() % (this->Buffer.size());
		auto res = this->Buffer[idx];
		this->Buffer.erase(this->Buffer.begin() + idx);
		return res;
	}
	else return Exp();
}

Exp ExpBuffer::sampleBatch(size_t sampleSize)
{
	std::vector<torch::Tensor> current(0);
	std::vector<torch::Tensor> action(0);
	std::vector<torch::Tensor> reward(0);
	std::vector<torch::Tensor> next(0);
	size_t SearchSize = (this->Buffer.size() > sampleSize) ? sampleSize : this->Buffer.size();

	for (auto i = 0; i < SearchSize; i++) {
		auto s = this->sample();
		current.push_back(s.current);
		action.push_back(s.action);
		reward.push_back(s.reward);
		next.push_back(s.next);
		//res.push_back(this->sample());
	}

	auto currentTensor = torch::vstack(current);
	auto actionTensor = torch::vstack(action);
	auto rewardTensor = torch::vstack(reward);
	auto nextTensor = torch::vstack(next);

	Exp BatchExp(currentTensor, actionTensor, rewardTensor, nextTensor);
	return BatchExp;
}
