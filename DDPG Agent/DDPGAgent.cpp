#include "DDPGAgent.h"

DDPGAgent& DDPGAgent::setActor(Actor actor, torch::Dtype dtype) {
	this->actor = actor;
	this->actorTarget = Actor();

	this->actorTarget->copyHardWeight(*(this->actor));

	return *this;
}
DDPGAgent& DDPGAgent::setCritic(Critic critic, torch::Dtype dtype) {
	this->critic = critic;
	this->criticTarget = Critic();

	this->criticTarget->copyHardWeight(*(this->critic));
	return *this;
}

DDPGAgent& DDPGAgent::setSoftUpdate(bool isSoftUpdate) {
	this->isSoftUpdate = isSoftUpdate;

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

void DDPGAgent::targetUpdate() {
	if (this->isSoftUpdate) {
		this->actorTarget->copySoftWeigth(*(this->actor), this->tau);
		this->criticTarget->copySoftWeight(*(this->critic), this->tau);
	}
	else {
		this->actorTarget->copyHardWeight(*(this->actor));
		this->criticTarget->copyHardWeight(*(this->critic));
	}
}

torch::Tensor DDPGAgent::act(torch::Tensor currentState) {
	if (this->isStochastic) {
		return this->actor->forward(currentState) + torch::randn_like(currentState) * sigma;
	}
	else {
		return this->actor->forward(currentState);
	}
}

void DDPGAgent::push(torch::Tensor currentState, torch::Tensor action, double reward, torch::Tensor nextState) {
	this->replayBuffer.push({ currentState, action, reward, nextState });
}

void DDPGAgent::train()
{
	//Optimizer
	auto aOptimizer = torch::optim::SGD(this->actor->parameters(), this->aAlpha);
	auto cOptimizer = torch::optim::SGD(this->critic->parameters(), this->cAlpha);
	for (int i = 0; i < epoch; i++) {
		auto sample = this->replayBuffer.sampleBatch(this->minibatchSize);
	}
}

void DDPGAgent::save() {
	//torch::save(this->actorTarget.getNetwork(), "Actor.pt");
	//torch::save(this->criticTarget.getNetwork(), "Critic.pt");
}