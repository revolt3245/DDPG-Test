#include <iostream>
#include <chrono>

#include "ExpBuffer.h"
#include "DDPGAgent.h"

using namespace std;

int main() {
	Actor actor;
	Critic critic;

	DDPGAgent agent = DDPGAgent()
		.setActor(actor)
		.setCritic(critic);

	auto aOptimizer = agent.getActorOptimizer();
	auto cOptimizer = agent.getCriticOptimizer();

	return 0;
}