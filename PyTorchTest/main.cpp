#include <iostream>
#include <fstream>
#include <torch/torch.h>

#include "Network.h"

using namespace std;

const unsigned int epoch = 1000;

int main() {
	//Dataset
	auto I = std::vector<torch::Tensor>(0);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			int T[] = { i, j };
			auto Ttensor = torch::from_blob(T, { 1, 2 }, torch::dtype(torch::kInt32));
			auto Trand = torch::randint(10, { 1,2 }, torch::dtype(torch::kInt32));
			Trand.copy_(Ttensor);
			I.push_back(Trand);

			//cout << I << endl;
		}
	}

	auto Idata = torch::vstack(I);
	Idata = Idata.to(torch::kFloat32);

	//cout << Idata << endl;

	auto T = vector<float>{ 0, 1, 1, 0 };
	auto O = torch::from_blob(T.data(), { 4,1 }, torch::dtype(torch::kFloat32));

	//Optimizer && Loss function
	Network n;

	torch::optim::SGD optimizer(n->parameters(), 0.1);
	torch::nn::MSELoss loss;

	//train
	for (auto i = 0; i < epoch; i++) {
		optimizer.zero_grad();
		auto pred = n->forward(Idata);
		auto cost = loss->forward(pred, O);
		cost.backward();
		optimizer.step();

		if ((i + 1) % 100 == 0) {
			cout << "Epoch : " << (i + 1) << ", cost : " << cost.data_ptr<float>()[0] << endl;
		}
	}

	//result
	cout << n->forward(Idata) << endl;


	return 0;
}