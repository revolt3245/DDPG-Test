#include "DDPGAgent.h"

extern torch::optim::SGD aOptimizer;
extern torch::optim::SGD cOptimizer;

DDPGAgent& DDPGAgent::setActor(Actor actor, torch::Dtype dtype) {
	this->actor = actor;
	this->actorTarget = Actor();

	this->actor->to(dtype);
	this->actorTarget->to(dtype);

	this->actorTarget->copyHardWeight(*(this->actor));

	return *this;
}
DDPGAgent& DDPGAgent::setCritic(Critic critic, torch::Dtype dtype) {
	this->critic = critic;
	this->criticTarget = Critic();

	this->critic->to(dtype);
	this->criticTarget->to(dtype);

	this->criticTarget->copyHardWeight(*(this->critic));
	return *this;
}

DDPGAgent& DDPGAgent::setStochastic(bool isStochastic) {
	this->isStochastic = isStochastic;

	return *this;
}

DDPGAgent& DDPGAgent::setDevice() {
	auto device = (torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;

	this->actor->to(device);
	this->actorTarget->to(device);
	this->critic->to(device);
	this->criticTarget->to(device);

	return *this;
}

torch::Tensor DDPGAgent::act(torch::Tensor currentState) {
	auto action = this->actor->forward(currentState);
	if (this->isStochastic) {
		return torch::clamp(10 * action + torch::randn_like(action) * sigma, -10, 10);
	}
	else {
		return torch::clamp(10 * action, -10, 10);
	}
}

void DDPGAgent::push(torch::Tensor currentState, torch::Tensor action, torch::Tensor reward, torch::Tensor nextState) {
	this->replayBuffer.push({ currentState, action, reward, nextState });
}

void DDPGAgent::train(torch::optim::SGD& aOptimizer, torch::optim::SGD& cOptimizer)
{
	for (int i = 0; i < epoch; i++) {
		if (this->replayBuffer.getSize() < this->minibatchSize)break;
		cout << "minibatch " << (i + 1) << endl;
		auto sample = this->replayBuffer.sampleBatch(this->minibatchSize);

		//critic train
		cOptimizer.zero_grad();
		auto actionNext = this->actorTarget->forward(sample.next);
		auto criticNext = this->criticTarget->forward(sample.next, actionNext);
		auto y = sample.reward + this->gamma * criticNext;
		auto criticCurrent = this->critic->forward(sample.current, sample.action);

		auto cCost = this->cLoss->forward(criticCurrent, y);
		cCost.backward();
		cOptimizer.step();

		//actor train
		aOptimizer.zero_grad();
		auto actionPred = this->actor->forward(sample.current);
		auto aCost = -this->critic->forward(sample.current, actionPred).mean();
		aCost.backward();
		aOptimizer.step();

		//soft update target
		cout << "critic cost" << endl;
		cout << cCost << endl;
		cout << "actor cost" << endl;
		cout << aCost << endl;

	}
	this->targetSoftUpdate();
	this->sigma =(this->sigma >= 1)?this->sigma * 0.95:this->sigma;
}

void DDPGAgent::save() {
	//torch::save(this->actorTarget.getNetwork(), "Actor.pt");
	//torch::save(this->criticTarget.getNetwork(), "Critic.pt");
}

torch::optim::SGD DDPGAgent::getActorOptimizer()
{
	return torch::optim::SGD(this->actor->parameters(), this->aAlpha);
}

torch::optim::SGD DDPGAgent::getCriticOptimizer()
{
	return torch::optim::SGD(this->critic->parameters(), this->cAlpha);
}

void DDPGAgent::targetSoftUpdate()
{
	this->actorTarget->copySoftWeigth(*(this->actor), this->tau);
	this->criticTarget->copySoftWeight(*(this->critic), this->tau);
}

void DDPGAgent::targetHardUpdate()
{
	this->actorTarget->copyHardWeight(*(this->actor));
	this->criticTarget->copyHardWeight(*(this->critic));
}
