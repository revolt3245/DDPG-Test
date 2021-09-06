#include <iostream>

#include <torch/torch.h>

using namespace std;

int main() {
	auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

	try {
		auto T = torch::randn({ 10000,4 }, torch::kFloat64);

		T = T.to(device);
		cout << T << endl;

		auto idx = torch::randperm(10000);

		cout << idx << endl;
	}
	catch (const c10::NotImplementedError& e) {
		cout << e.what() << endl;
	}

	return 0;
}