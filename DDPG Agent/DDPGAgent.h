#pragma once
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

#include "Actor.h"
#include "Critic.h"

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
		isSoftUpdate(false)
	{};

	DDPGAgent& setActor(Actor Actor);
	DDPGAgent& setCritic(Critic Critic);

	DDPGAgent& setDevice();

	void targetUpdate();

	void save();
protected:
	void copyHardWeight(shared_ptr<torch::nn::Module> target, shared_ptr<torch::nn::Module> source);
	void copySoftWeight(shared_ptr<torch::nn::Module> target, shared_ptr<torch::nn::Module> source);
private:
	Actor actor;
	Critic critic;

	Actor actorTarget;
	Critic criticTarget;

	double gamma;
	double aAlpha;
	double cAlpha;

	double tau;

	bool isSoftUpdate;
};

