#pragma once
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

#include "Actor.h"
#include "Critic.h"
#include "ExpBuffer.h"

using namespace std;

class DDPGAgent
{
public:
	DDPGAgent()
		:actor(), critic(),
		actorTarget(), criticTarget(),
		gamma(0.99),
		aAlpha(1e-3), cAlpha(5e-4),
		tau(0.001),
		isStochastic(false),
		sigma(0.2),
		minibatchSize(256), epoch(1),
		replayBuffer(1e6),
		cLoss()
	{
	};

	DDPGAgent& setActor(Actor Actor, torch::Dtype dtype);
	DDPGAgent& setCritic(Critic Critic, torch::Dtype dtype);

	DDPGAgent& setStochastic(bool isStochastic);

	DDPGAgent& setDevice();

	torch::Tensor act(torch::Tensor currentState);
	void push(torch::Tensor currentState, torch::Tensor action, torch::Tensor reward, torch::Tensor nextState, torch::Tensor terminateTensor);

	size_t getBufferSize();
	
	void train(torch::optim::SGD& aOptimizer, torch::optim::SGD& cOptimizer, torch::Device device);
	void train(torch::optim::Adam& aOptimizer, torch::optim::Adam& cOptimizer, torch::Device device);

	void noiseDiminish(double rate, double threshold);

	void eval();
	void trainMode();

	void save();

	double getSigma();

	double getQ0();

	torch::optim::SGD getActorOptimizer();
	torch::optim::SGD getCriticOptimizer();

	torch::optim::Adam getActorOptimizerAdam();
	torch::optim::Adam getCriticOptimizerAdam();

	torch::optim::RMSprop getActorOptimizerRMS();
	torch::optim::RMSprop getCriticOptimizerRMS();

	void targetSoftUpdate();
	void targetHardUpdate();
private:
	Actor actor;
	Critic critic;

	Actor actorTarget;
	Critic criticTarget;

	ExpBuffer replayBuffer;

	double gamma;
	double aAlpha;
	double cAlpha;

	double tau;

	bool isStochastic;

	double sigma;

	size_t minibatchSize;
	int epoch;

	torch::nn::MSELoss cLoss;
};

