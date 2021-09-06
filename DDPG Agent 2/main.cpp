#include <iostream>
#include <chrono>
#include <fstream>

#include <MatlabEngine.hpp>
#include <MatlabDataArray.hpp>

#include "ExpBuffer.h"
#include "DDPGAgent.h"
#include "OUNoise.h"

using namespace std;

const unsigned int episode_size = 100000;
const unsigned int max_step = 500;
const unsigned int start_episode = 256;

const unsigned int movingAverageWindow = 5;

queue<int> Queue;

bool terminate_flag = false;

int main() {
	srand(time(0));
	std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = matlab::engine::startMATLAB();
	matlab::data::ArrayFactory arrayFactory;

	ofstream file("OUNoise_Episodes.txt");

	auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
	Actor actor;
	Critic critic;

	DDPGAgent agent = DDPGAgent()
		.setActor(actor, torch::kFloat64)
		.setCritic(critic, torch::kFloat64)
		.setStochastic(true)
		.setDevice();

	OUNoise N;
	N.setMu(torch::zeros({ 1,1 }, torch::TensorOptions().device(device).dtype(torch::kFloat64)));

	auto aOptimizer = agent.getActorOptimizerAdam();
	auto cOptimizer = agent.getCriticOptimizerAdam();

	matlabPtr->eval(u"Env = CartPoleContinuousAction2;");
	matlabPtr->eval(u"plot(Env);");

	auto env = matlabPtr->getVariable("Env");

	double movingAverage = 0;

	//learning
	for (int i = 0; i < episode_size; i++) {
		int current_step;
		int top;
		agent.eval();

		N.reset(torch::zeros({ 1, 1 }, torch::TensorOptions().device(device).dtype(torch::kFloat64)));
		cout << "episode " << (i + 1) << " ";
		matlab::data::TypedArray<double> state = matlabPtr->feval("reset", { env });
		for (int j = 0; j < max_step; j++) {
			auto statePtr = vector<double>(0);
			for (auto t = state.begin(); t != state.end(); t++) {
				statePtr.push_back(*t);
			}
			auto StateTorch = torch::zeros({ 1,4 }, torch::TensorOptions().dtype(torch::kFloat64).device(device))
				.copy_(torch::from_blob(statePtr.data(), { 1,4 }, torch::kFloat64));
			auto action = agent.act(StateTorch);
			if (agent.getBufferSize() < start_episode) action = torch::randn_like(action).clip_(-1, 1);
			auto actionScaled = 10 * action;

			auto nextStates = matlabPtr->feval("step", 4, { env, arrayFactory.createArray({1,1}, {actionScaled.cpu().data<double>()[0] }) });
			state = nextStates[0];
			matlab::data::TypedArray<double> reward = nextStates[1];
			matlab::data::TypedArray<bool> isDone = nextStates[2];

			auto nextPtr = vector<double>(0);
			for (auto t = state.begin(); t != state.end(); t++) {
				nextPtr.push_back(*t);
			}
			auto rewardPtr = vector<double>(0);
			rewardPtr.push_back(*(reward.begin()));

			auto terminatePtr = vector<double>(0);
			terminatePtr.push_back(1 - (double)*(isDone.begin()));

			auto rewardTorch = torch::zeros({ 1,1 }, torch::TensorOptions().dtype(torch::kFloat64).device(device))
				.copy_(torch::from_blob(rewardPtr.data(), { 1,1 }, torch::kFloat64));
			auto nextTorch = torch::zeros({ 1,4 }, torch::TensorOptions().dtype(torch::kFloat64).device(device))
				.copy_(torch::from_blob(nextPtr.data(), { 1,4 }, torch::kFloat64));
			auto terminateTorch = torch::zeros({ 1,1 }, torch::TensorOptions().dtype(torch::kFloat64).device(device))
				.copy_(torch::from_blob(terminatePtr.data(), { 1,1 }, torch::kFloat64));

			/*
			auto rewardTorch = torch::from_blob(rewardPtr.data(), { 1,1 }, torch::kFloat64).detach_();
			auto nextTorch = torch::from_blob(nextPtr.data(), { 1,4 }, torch::kFloat64).detach_();
			auto terminateTorch = torch::from_blob(terminatePtr.data(), { 1, 1 }, torch::kFloat64).detach_();
			*/

			agent.push(StateTorch, action, rewardTorch, nextTorch, terminateTorch);

			if (*(isDone.begin())) {
				cout << "total step : " << (j + 1) << " Q0 : " << agent.getQ0() << endl;
				current_step = j + 1;
				break;
			}
			else if (j + 1 >= max_step) {
				cout << "total step : " << (j + 1) << " Q0 : " << agent.getQ0() << endl;
				current_step = j + 1;
			}
		}

		file << current_step << endl;

		if (Queue.size() < movingAverageWindow) {
			top = 0;
		}
		else {
			top = Queue.front();
			Queue.pop();
		}
		Queue.push(current_step);
		movingAverage += ((double)current_step - top) / movingAverageWindow;

		if (movingAverage > 498)break;
		
		if ((i + 1) % 10 == 0) {
			//cout << "Noise : " << agent.getSigma() << endl;
			cout << "Trained Model ";
			agent.setStochastic(false);

			matlab::data::TypedArray<double> state = matlabPtr->feval("reset", { env });
			agent.eval();
			for (int i = 0; i < max_step; i++) {
				auto statePtr = vector<double>(0);
				for (auto t = state.begin(); t != state.end(); t++) {
					statePtr.push_back(*t);
				}
				auto StateTorch = torch::zeros({ 1,4 }, torch::TensorOptions().dtype(torch::kFloat64).device(device)).copy_(torch::from_blob(statePtr.data(), { 1,4 }, torch::kFloat64));
				auto action = agent.act(StateTorch);
				auto actionScaled = 10 * action;

				auto nextStates = matlabPtr->feval("step", 4, { env, arrayFactory.createArray({1,1}, {actionScaled.cpu().data<double>()[0] }) });
				state = nextStates[0];
				matlab::data::TypedArray<double> reward = nextStates[1];
				matlab::data::TypedArray<bool> isDone = nextStates[2];

				auto nextPtr = vector<double>(0);
				for (auto t = state.begin(); t != state.end(); t++) {
					nextPtr.push_back(*t);
				}
				auto rewardPtr = vector<double>(0);
				rewardPtr.push_back(*(reward.begin()));

				auto rewardTorch = torch::zeros({ 1,1 }, torch::TensorOptions().dtype(torch::kFloat64).device(device))
					.copy_(torch::from_blob(rewardPtr.data(), { 1,1 }, torch::kFloat64));
				auto nextTorch = torch::zeros({ 1,4 }, torch::TensorOptions().dtype(torch::kFloat64).device(device))
					.copy_(torch::from_blob(nextPtr.data(), { 1,4 }, torch::kFloat64));

				if (*(isDone.begin())) {
					cout << "total step : " << (i + 1) << endl;
					break;
				}
			}

			agent.setStochastic(true);
		}

		if (agent.getBufferSize() >= start_episode) {
			agent.trainMode();
			agent.train(aOptimizer, cOptimizer, device);
			agent.targetSoftUpdate();
		}
	}
	agent.save();
	agent.setStochastic(false);
	matlabPtr->eval(u"plot(Env);");
	//testing
	matlab::data::TypedArray<double> state = matlabPtr->feval("reset", { env });
	agent.eval();
	try {
		for (int i = 0; i < max_step; i++) {
			auto statePtr = vector<double>(0);
			for (auto t = state.begin(); t != state.end(); t++) {
				statePtr.push_back(*t);
			}
			auto StateTorch = torch::zeros({ 1,4 }, torch::TensorOptions().dtype(torch::kFloat64).device(device)).copy_(torch::from_blob(statePtr.data(), { 1,4 }, torch::kFloat64));
			auto action = agent.act(StateTorch);
			auto actionScaled = 10 * action;

			auto nextStates = matlabPtr->feval("step", 4, { env, arrayFactory.createArray({1,1}, {actionScaled.cpu().data<double>()[0] }) });
			state = nextStates[0];
			matlab::data::TypedArray<double> reward = nextStates[1];
			matlab::data::TypedArray<bool> isDone = nextStates[2];

			if (*(isDone.begin())) {
				cout << "total step : " << (i + 1) << endl;
				break;
			}
		}
	}
	catch (const c10::Error& e) {
		cout << e.what() << endl;
	}
	return 0;
}