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
	os << "terminate" << endl;
	os << exp.terminate << endl;

	return os;
}

Exp& Exp::operator=(const Exp& obj) {
	this->current = obj.current.detach();
	this->action = obj.action.detach();
	this->reward = obj.reward.detach();
	this->next = obj.next.detach();
	this->terminate = obj.terminate.detach();

	return *this;
}
Exp Exp::cpu()
{
	Exp res;
	res.current = this->current.detach().cpu();
	res.action = this->action.detach().cpu();
	res.reward = this->reward.detach().cpu();
	res.next = this->next.detach().cpu();
	res.terminate = this->terminate.detach().cpu();
	return res;
}
Exp Exp::cuda()
{
	Exp res;
	res.current = this->current.detach().cuda();
	res.action = this->action.detach().cuda();
	res.reward = this->reward.detach().cuda();
	res.next = this->next.detach().cuda();
	res.terminate = this->terminate.detach().cuda();

	return res;
}
Exp Exp::to(torch::Device device)
{
	Exp res;
	res.current = this->current.to(device);
	res.action = this->action.to(device);
	res.reward = this->reward.to(device);
	res.next = this->next.to(device);
	res.terminate = this->terminate.to(device);

	return res;
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
	Buffer.push_back(experience.cpu());
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
		return res;
	}
	else return Exp();
}

Exp ExpBuffer::sample(unsigned int idx)
{
	if (this->Buffer.size() > 0) {
		auto res = this->Buffer[idx];
		return res;
	}
	else return Exp();
}

Exp ExpBuffer::sample(torch::Tensor indices)
{
	if (this->Buffer.size() > 0) {
		auto indicesSize = indices.size(-1);
		auto indicesPtr = indices.data<int>();

		std::vector<torch::Tensor> current(0);
		std::vector<torch::Tensor> action(0);
		std::vector<torch::Tensor> reward(0);
		std::vector<torch::Tensor> next(0);
		std::vector<torch::Tensor> terminate(0);

		for (auto i = 0; i < indicesSize; i++) {
			auto s = this->sample(indicesPtr[i]);

			current.push_back(s.current);
			action.push_back(s.action);
			reward.push_back(s.reward);
			next.push_back(s.next);
			terminate.push_back(s.terminate);
		}

		auto currentTensor = torch::vstack(current).detach();
		auto actionTensor = torch::vstack(action).detach();
		auto rewardTensor = torch::vstack(reward).detach();
		auto nextTensor = torch::vstack(next).detach();
		auto terminateTensor = torch::vstack(terminate).detach();

		Exp BatchExp(currentTensor, actionTensor, rewardTensor, nextTensor, terminateTensor);

		//cout << BatchExp << endl;
		return BatchExp;
	}
	else return Exp();
}

Exp ExpBuffer::sampleBatch(size_t sampleSize)
{
	std::vector<torch::Tensor> current(0);
	std::vector<torch::Tensor> action(0);
	std::vector<torch::Tensor> reward(0);
	std::vector<torch::Tensor> next(0);
	std::vector<torch::Tensor> terminate(0);

	auto maxSize = (sampleSize > this->Buffer.size()) ? this->Buffer.size() : sampleSize;
	for (auto i = 0; i < sampleSize; i++) {
		auto s = this->sample();
		current.push_back(s.current);
		action.push_back(s.action);
		reward.push_back(s.reward);
		next.push_back(s.next);
		terminate.push_back(s.terminate);
	}

	//cout << endl;

	auto currentTensor = torch::vstack(current).detach();
	auto actionTensor = torch::vstack(action).detach();
	auto rewardTensor = torch::vstack(reward).detach();
	auto nextTensor = torch::vstack(next).detach();
	auto terminateTensor = torch::vstack(terminate).detach();

	Exp BatchExp(currentTensor, actionTensor, rewardTensor, nextTensor, terminateTensor);

	//cout << BatchExp << endl;
	return BatchExp;
}

Exp ExpBuffer::sampleBatch(size_t sampleSize, torch::Device device)
{
	/*
	std::vector<torch::Tensor> current(0);
	std::vector<torch::Tensor> action(0);
	std::vector<torch::Tensor> reward(0);
	std::vector<torch::Tensor> next(0);
	std::vector<torch::Tensor> terminate(0);

	auto maxSize = (sampleSize > this->Buffer.size()) ? this->Buffer.size() : sampleSize;

	//auto idxes = torch::randint(this->Buffer.size(), { (long long)maxSize });

	for (auto i = 0; i < sampleSize; i++) {
		auto s = this->sample();
		current.push_back(s.current);
		action.push_back(s.action);
		reward.push_back(s.reward);
		next.push_back(s.next);
		terminate.push_back(s.terminate);
	}

	auto currentTensor = torch::vstack(current).detach();
	auto actionTensor = torch::vstack(action).detach();
	auto rewardTensor = torch::vstack(reward).detach();
	auto nextTensor = torch::vstack(next).detach();
	auto terminateTensor = torch::vstack(terminate).detach();

	Exp BatchExp(currentTensor, actionTensor, rewardTensor, nextTensor, terminateTensor);

	//cout << BatchExp << endl;
	*/

	auto maxSize = (sampleSize > this->Buffer.size()) ? this->Buffer.size() : sampleSize;
	auto indices = torch::randperm(this->Buffer.size(), torch::kInt32).slice(-1, 0, maxSize);

	auto BatchExp = this->sample(indices);

	return BatchExp.to(device);
}
