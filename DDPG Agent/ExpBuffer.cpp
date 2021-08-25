#include "ExpBuffer.h"

size_t ExpBuffer::getMaxSize()
{
	return this->Size;
}

void ExpBuffer::setMaxSize(size_t Size)
{
	this->Size = Size;
}

size_t ExpBuffer::getSize()
{
	return this->Buffer.size();
}

void ExpBuffer::push(Exp experience)
{
	if (this->Buffer.size() >= this->Size) {
		Buffer.erase(this->Buffer.begin());
	}
	Buffer.push_back(experience);
}

bool ExpBuffer::isEmpty()
{
	return this->Buffer.empty();
}

Exp ExpBuffer::sample()
{
	if (this->Buffer.size() > 0) {
		srand(time(0));

		auto idx = rand() % (this->Buffer.size());
		auto res = this->Buffer[idx];
		this->Buffer.erase(this->Buffer.begin() + idx);
		return res;
	}
	else return Exp();
}

std::vector<Exp> ExpBuffer::sampleBatch(size_t sampleSize)
{
	std::vector<Exp> res(0);
	size_t SearchSize = (this->Buffer.size() > sampleSize) ? sampleSize : this->Buffer.size();

	for (auto i = 0; i < SearchSize; i++) {
		res.push_back(this->sample());
	}
	return res;
}
