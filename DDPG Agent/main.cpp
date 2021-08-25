#include <iostream>
#include <torch/torch.h>
#include <chrono>

#include "DDPGAgent.h"

int main() {
	auto device = (torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;

	Actor actor;
	Critic critic;

	for (int i = 0; i < 10000; i++) {
		torch::Tensor Test = torch::randn({ 1, 6 });
		auto start = std::chrono::high_resolution_clock::now();
		actor->forward(Test);
		auto end = std::chrono::high_resolution_clock::now();

		auto elapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

		cout << elapse << endl;
	}
	return 0;
}