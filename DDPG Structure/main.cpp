#include <iostream>
#include <chrono>
#include <fstream>

#include <ATen/ATen.h>

#include <torch/torch.h>
#include <torch/cuda.h>

#include "Actor.h"
#include "Critic.h"

int main() {
	auto device = (torch::cuda::is_available())?torch::kCUDA:torch::kCPU;

	auto actor = Actor();
	std::ofstream datafile("elapse.csv");

	actor.to(device);

	torch::Tensor x = torch::randn({ 1, 8 });
	torch::Tensor res;
	const unsigned int trial = 10000;
	unsigned int total = 0;

	unsigned int max = 0;
	unsigned int min = -1;

	for (auto i = 0; i < trial; i++) {
		x = torch::randn({ 1, 6 });
		auto start = std::chrono::high_resolution_clock::now();
		res = actor.forward(x);
		auto end = std::chrono::high_resolution_clock::now();

		auto elapse = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

		datafile << elapse << "\n";

		total += elapse;

		max = (max < elapse) ? elapse : max;
		min = (min > elapse) ? elapse : min;
	}

	std::cout << "Trial : " << trial << std::endl;
	std::cout << "mean : " << total / trial << " us" << std::endl;
	std::cout << "best : " << min << " us" << std::endl;
	std::cout << "worst : " << max << " us" << std::endl;

	return 0;
}