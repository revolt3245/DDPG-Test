#include "ExpBuffer.h"

ostream& operator<<(ostream& os, Exp exp) {
	os << "current" << endl;
	os << exp.current << endl;
	os << "action" << endl;
	os << exp.action << endl;
	os << "reward" << endl;
	os << exp.reward << endl;
	os << "next" << endl;
	os << exp.next << endl;

	return os;
}

Exp& Exp::operator=(const Exp& obj) {
	this->current = torch::zeros_like(obj.current, obj.current.dtype()).copy_(obj.current);
	this->action = torch::zeros_like(obj.action, obj.current.dtype()).copy_(obj.action);
	this->reward = torch::zeros_like(obj.reward, obj.current.dtype()).copy_(obj.reward);
	this->next = torch::zeros_like(obj.next, obj.current.dtype()).copy_(obj.next);

	return *this;
}
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

	srand(time(0));

	auto maxSize = (sampleSize > this->Buffer.size()) ? this->Buffer.size() : sampleSize;
	for (auto i = 0; i < maxSize; i++) {
		auto s = this->sample();
		current.push_back(s.current);
		action.push_back(s.action);
		reward.push_back(s.reward);
		next.push_back(s.next);
	}

	auto currentTensor = torch::vstack(current);
	auto actionTensor = torch::vstack(action);
	auto rewardTensor = torch::vstack(reward);
	auto nextTensor = torch::vstack(next);
	Exp BatchExp(currentTensor, actionTensor, rewardTensor, nextTensor);
	return BatchExp;
}
