#include <iostream>
#include <torch/torch.h>

using namespace std;

int main() {
	auto T = torch::randn({ 1, 6 });
	return 0;
}