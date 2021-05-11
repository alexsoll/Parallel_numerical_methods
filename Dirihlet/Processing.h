#pragma once
#include <iostream>

using namespace std;

void PrintArray(int size, double* a) {
	std::cout << "( ";
	for (int i = 0; i < size; i++) {
		std::cout << a[i] << " ";
	}
	std::cout << ")" << std::endl;
}

class heat_task {
public:
    double X = 1;
    double Y = 1;
    int n = 4;
    int m = 4;
    double left_condition(double y) { return 1.0 + y * y; }
    double right_condition(double y) { return 4.0 + y * y; }
    double bottom_condition(double x) { return 4.0 + x * x; }
    double top_condition(double x) { return 9.0 + x * x; }
    double f(double x, double y) { return 4.0; }
};