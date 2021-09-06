#include "DDPGAgent.h"

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
	//cout << action.cpu().data<double>()[0] << endl;
	if (this->isStochastic) {
		/*
		vector<double> Gain = { -0.3162, -0.4157, -5.9616, -0.6582 };
		auto mbGain = torch::from_blob(Gain.data(), { 4, 1 }, torch::kFloat64);
		auto mbAction = torch::matmul(currentState, mbGain.cuda());
		auto randomProcess = torch::randn_like(mbAction, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
		*/
		//return torch::clamp(action - mbAction * this->sigma * 0.5 + randomProcess * 0.33, -1, 1);

		return torch::clamp(action + this->sigma * torch::randn_like(action), -1, 1);
	}
	else {
		return torch::clamp(action, -1, 1);
	}
}

void DDPGAgent::push(torch::Tensor currentState, torch::Tensor action, torch::Tensor reward, torch::Tensor nextState, torch::Tensor terminate)
{
	this->replayBuffer.push({ currentState, action, reward, nextState, terminate });
}

size_t DDPGAgent::getBufferSize()
{
	return this->replayBuffer.getSize();
}

void DDPGAgent::train(torch::optim::SGD& aOptimizer, torch::optim::SGD& cOptimizer, torch::Device device)
{
	for (int i = 0; i < epoch; i++) {
		//cout << "minibatch " << (i + 1) << endl;
		auto sample = this->replayBuffer.sampleBatch(this->minibatchSize, device);

		//critic train
		cOptimizer.zero_grad();
		auto actionNext = this->actorTarget->forward(sample.next);
		auto criticNext = this->criticTarget->forward(sample.next, actionNext);
		auto y = sample.reward + this->gamma * criticNext * sample.terminate;
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
	}
}

void DDPGAgent::train(torch::optim::Adam& aOptimizer, torch::optim::Adam& cOptimizer, torch::Device device)
{
	for (int i = 0; i < epoch; i++) {
		//cout << "minibatch " << (i + 1) << endl;
		auto sample = this->replayBuffer.sampleBatch(this->minibatchSize, device);

		//critic train
		cOptimizer.zero_grad();
		auto actionNext = this->actorTarget->forward(sample.next);
		auto criticNext = this->criticTarget->forward(sample.next, actionNext);
		auto y = sample.reward + this->gamma * criticNext * sample.terminate;
		auto criticCurrent = this->critic->forward(sample.current, sample.action);

		auto cCost = this->cLoss->forward(criticCurrent, y);
		cCost.backward();
		cOptimizer.step();

		//actor train
		aOptimizer.zero_grad();
		auto actionPred = this->actor->forward(sample.current);
		auto aCost = this->critic->forward(sample.current, actionPred).mean();
		aCost = -aCost;
		aCost.backward();
		//cout << this->actor->parameters()[2].grad() << endl;
		aOptimizer.step();
	}
}

void DDPGAgent::noiseDiminish(double rate, double threshold)
{
	this->sigma = (this->sigma <= threshold) ? threshold : this->sigma * rate;
}

void DDPGAgent::eval()
{
	this->actor->eval();
	this->actorTarget->eval();
	this->critic->eval();
	this->criticTarget->eval();
}

void DDPGAgent::trainMode()
{
	this->actor->train();
	this->actorTarget->train();
	this->critic->train();
	this->criticTarget->train();
}

void DDPGAgent::save() {
	torch::save(this->actor, "Actor.pt");
	torch::save(this->critic, "Critic.pt");
}

double DDPGAgent::getSigma()
{
	return this->sigma;
}

double DDPGAgent::getQ0()
{
	auto s = torch::zeros({ 1,4 }, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));

	auto Q0 = this->critic->forward(s, this->actor->forward(s));

	return Q0.cpu().data<double>()[0];
}

torch::optim::SGD DDPGAgent::getActorOptimizer()
{
	return torch::optim::SGD(this->actor->parameters(), this->aAlpha);
}

torch::optim::SGD DDPGAgent::getCriticOptimizer()
{
	return torch::optim::SGD(this->critic->parameters(), this->cAlpha);
}

torch::optim::Adam DDPGAgent::getActorOptimizerAdam()
{
	return torch::optim::Adam(this->actor->parameters(), this->aAlpha);
}

torch::optim::Adam DDPGAgent::getCriticOptimizerAdam()
{
	return torch::optim::Adam(this->critic->parameters(), this->cAlpha);
}

torch::optim::RMSprop DDPGAgent::getActorOptimizerRMS()
{
	return torch::optim::RMSprop(this->actor->parameters(), this->aAlpha);
}

torch::optim::RMSprop DDPGAgent::getCriticOptimizerRMS()
{
	return torch::optim::RMSprop(this->critic->parameters(), this->cAlpha);
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
