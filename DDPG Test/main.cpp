#include <iostream>

using namespace std;

float bias = 55000 / 65535;

int main() {
	float x;
	unsigned short y;

	while (true) {
		cin >> x;
		if (x < 0.001) {
			y = (unsigned short)(0.999 * 65535);
		}
		else if (x < 0.999) {
			y = (unsigned short)((1 - x) * 65535);
		}
		else {
			y = (unsigned short)(0.001 * 65535);
		}
		cout << y << endl;
	}
	return 0;
}