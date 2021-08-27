#include <iostream>
#include <chrono>

#include <MatlabEngine.hpp>
#include <MatlabDataArray.hpp>

#include "ExpBuffer.h"
#include "DDPGAgent.h"

using namespace std;

const unsigned int episode_size = 10000;
const unsigned int max_step = 1000;

int main() {
	std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = matlab::engine::startMATLAB();
	matlab::data::ArrayFactory arrayFactory;
	Actor actor;
	Critic critic;

	DDPGAgent agent = DDPGAgent()
		.setActor(actor, torch::kFloat64)
		.setCritic(critic, torch::kFloat64)
		.setStochastic(true);

	auto aOptimizer = agent.getActorOptimizer();
	auto cOptimizer = agent.getCriticOptimizer();

	matlabPtr->eval(u"Env = rlPredefinedEnv(\"CartPole-Continuous\");");
	matlabPtr->eval(u"Env.PenaltyForFalling = -5;");
	matlabPtr->eval(u"plot(Env);");

	auto env = matlabPtr->getVariable("Env");

	//learning
	for (int i = 0; i < episode_size; i++) {
		cout << "episode " << (i + 1) << " ";
		matlab::data::TypedArray<double> state = matlabPtr->feval("reset", { env });
		for (int j = 0; j < max_step; j++) {
			//auto statePtr = state.release().release();
			auto statePtr = vector<double>(0);
			for (auto t = state.begin(); t != state.end(); t++) {
				statePtr.push_back(*t);
			}
			auto StateTorch = torch::zeros({ 1,4 }, torch::kFloat64).copy_(torch::from_blob(statePtr.data(), { 1,4 }, torch::kFloat64));
			auto action = agent.act(StateTorch);

			action.data<double>();

			auto nextStates = matlabPtr->feval("step", 4, { env, arrayFactory.createArray({1,1}, {action.data<double>()[0] }) });
			state = nextStates[0];
			matlab::data::TypedArray<double> reward = nextStates[1];
			matlab::data::TypedArray<bool> isDone = nextStates[2];

			auto nextPtr = vector<double>(0);
			for (auto t = state.begin(); t != state.end(); t++) {
				nextPtr.push_back(*t);
			}
			auto rewardPtr = vector<double>(0);
			rewardPtr.push_back(*(reward.begin()));

			auto rewardTorch = torch::zeros({ 1,1 }, torch::kFloat64).copy_(torch::from_blob(rewardPtr.data(), { 1,1 }, torch::kFloat64));
			auto nextTorch = torch::zeros({ 1,4 }, torch::kFloat64).copy_(torch::from_blob(nextPtr.data(), { 1,4 }, torch::kFloat64));

			agent.push(StateTorch, action / 10, rewardTorch, nextTorch);

			if (*(isDone.begin())) {
				cout << "total step : " << (j + 1) << endl;
				break;
			}
			else if (j + 1 >= 500) {
				cout << "total step : " << (j + 1) << endl;
			}
		}

		if((i+1)%10 == 0)agent.train(aOptimizer, cOptimizer);
	}
	agent.setStochastic(false);
	//testing
	matlab::data::TypedArray<double> state = matlabPtr->feval("reset", { env });
	for (int i = 0; i < max_step; i++) {
		auto statePtr = vector<double>(0);
		for (auto t = state.begin(); t != state.end(); t++) {
			statePtr.push_back(*t);
		}
		auto StateTorch = torch::zeros({ 1,4 }, torch::kFloat64).copy_(torch::from_blob(statePtr.data(), { 1,4 }, torch::kFloat64));
		auto action = agent.act(StateTorch);

		action.data<double>();

		auto nextStates = matlabPtr->feval("step", 4, { env, arrayFactory.createArray({1,1}, {action.data<double>()[0] }) });
		state = nextStates[0];
		matlab::data::TypedArray<double> reward = nextStates[1];
		matlab::data::TypedArray<bool> isDone = nextStates[2];

		if (*(isDone.begin())) {
			cout << "total step : " << (i + 1) << endl;
		}
	}
	return 0;
}