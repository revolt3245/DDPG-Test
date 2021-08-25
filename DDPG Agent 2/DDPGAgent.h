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
		aAlpha(1e-3), cAlpha(1e-3),
		tau(0.01),
		isStochastic(false),
		sigma(1.0),
		minibatchSize(256), epoch(10),
		aLoss(), cLoss()
	{
	};

	DDPGAgent& setActor(Actor Actor);
	DDPGAgent& setCritic(Critic Critic);

	DDPGAgent& setStochastic(bool isStochastic);

	DDPGAgent& setDevice();

	torch::Tensor act(torch::Tensor currentState);
	void push(torch::Tensor currentState, torch::Tensor action, torch::Tensor reward, torch::Tensor nextState);

	void train(torch::optim::SGD aOptimizer, torch::optim::SGD cOptimizer);

	void save();

	torch::optim::SGD getActorOptimizer();
	torch::optim::SGD getCriticOptimizer();
protected:
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

	torch::nn::MSELoss aLoss;
	torch::nn::MSELoss cLoss;
};

