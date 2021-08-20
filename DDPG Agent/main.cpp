#include <iostream>
#include <torch/torch.h>

#include "DDPGAgent.h"
int main() {
	auto device = (torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;

	Actor actor;
	Critic critic;

	actor.to(device);
	critic.to(device);

	auto agent = DDPGAgent()
		.setActor(actor)
		.setCritic(critic)
		;

	agent.save();
	return 0;
}