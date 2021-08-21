#include "DDPGAgent.h"

DDPGAgent& DDPGAgent::setActor(Actor actor) {
	this->actor = actor;
	this->actorTarget = Actor();

	this->actorTarget->copyHardWeight(*(this->actor));

	return *this;
}
DDPGAgent& DDPGAgent::setCritic(Critic critic) {
	this->critic = critic;
	this->criticTarget = Critic();

	this->criticTarget->copyHardWeight(*(this->critic));
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

void DDPGAgent::save() {
	//torch::save(this->actorTarget.getNetwork(), "Actor.pt");
	//torch::save(this->criticTarget.getNetwork(), "Critic.pt");
}