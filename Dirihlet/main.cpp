#include <iostream>
#include <vector>
#include "omp.h"
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;    

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

void heat_dirichlet_sor(heat_task task, double* v) {
    double h = task.X / task.n;
    double k = task.Y / task.m;

    double h_ = 1 / pow(h, 2);
    double k_ = 1 / pow(k, 2);

    double omega = 2 / (1 + 2 * sin(M_PI * h / 2));

    double D = 2 * (h_ + k_);

    double eps = 1e-5;

    vector<vector<double>> nu(task.n + 1, vector<double>(task.m + 1, 0.0));

    // Boundary
    for (int i = 0; i <= task.n; i++) {
        nu[i][0] = task.bottom_condition(i * h);
        nu[i][task.m] = task.top_condition(i * h);
    }
    for (int i = 0; i <= task.n; i++) {
        nu[0][i] = task.left_condition(i * k);
        nu[task.n][i] = task.right_condition(i * k);
    }


    // The function of external impact
    vector<vector<double>> f(task.n + 1, vector<double>(task.m + 1, 0.0));
    for (int i = 0; i <= task.n; i++) {
        for (int j = 0; j <= task.m; j++) {
            f[i][j] = task.f(i * h, j * k);
        }
    }

    int maxIter = 50;
    //double err;

    for (int iter = 0; iter < maxIter; iter++) {
        //err = 0.;
        for (int i = 1; i < task.n; i++) {
            for (int j = 1; j < task.m; j++) {                
                nu[i][j] = omega * h_ * nu[i - 1][j] + omega * k_ * nu[i][j - 1] + omega * h_ * nu[i + 1][j] + omega * k_ * nu[i][j + 1] + \
                    (1 - omega) * D * nu[i][j] + omega * f[i][j];
                nu[i][j] /= D;
                std::cout << "nu[" << i << "][" << j << "] = " << nu[i][j] << std::endl;
            }
        }
    }


    for (int i = 0; i <= task.n; i++)
    {
        for (int j = 0; j <= task.m; j++)
        {
            v[i * (task.m + 1) + j] = nu[i][j];
        }
    }
}

void PrintArray(int size, double* a) {
    std::cout << "( ";
    for (int i = 0; i < size; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    heat_task t;
    double* v = new double[(t.n + 1) * (t.m + 1)];
    std::cout << "Size of vector : " << (t.n + 1) * (t.m + 1) << std::endl;
    heat_dirichlet_sor(t, v);

    PrintArray((t.n + 1) * (t.m + 1), v);

    return 0;
}