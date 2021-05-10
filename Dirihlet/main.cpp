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

    double eps = 1e-15;

    vector<vector<double>> nu(task.n + 1, vector<double>(task.m + 1, 0.0));

    // Boundary
#pragma omp parallel for
    for (int i = 0; i <= task.n; i++) {
        nu[i][0] = task.bottom_condition(i * h);
        nu[i][task.m] = task.top_condition(i * h);
    }
#pragma omp parallel for
    for (int i = 0; i <= task.m; i++) {
        nu[0][i] = task.left_condition(i * k);
        nu[task.n][i] = task.right_condition(i * k);
    }

    // The function of external impact
    vector<vector<double>> f(task.n + 1, vector<double>(task.m + 1, 0.0));
#pragma omp parallel for
    for (int i = 0; i <= task.n; i++) {
        for (int j = 0; j <= task.m; j++) {
            f[i][j] = - task.f(i * h, j * k);
        }
    }

    int j;
    double tmp;
    double err;
    double prev;
    do {
        err = 0.;
        for (int k = 0; k < task.n + task.m - 3; k++) {
#pragma omp parallel for private(j, tmp, prev)
            for (int i = min(1 + k, task.n - 1); i >= max(1, k - task.m + 3); i--) {
                j = task.m - (k - i + 2);

                prev = nu[i][j];
                tmp = - D * prev + h_ * (nu[i - 1][j] + nu[i + 1][j]) + k_ * (nu[i][j - 1] + nu[i][j + 1]);

                nu[i][j] = prev + omega * (tmp + f[i][j]) / D;

                tmp = fabs(nu[i][j] - prev);

#pragma omp critical
                {
                    if (tmp > err) {
                        err = tmp;
                    }
                }
            }
        }
    } while (tmp > eps);

#pragma omp parallel for
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

    heat_dirichlet_sor(t, v);
    
    PrintArray((t.n + 1) * (t.m + 1), v);
    
    delete[] v;
    return 0;
}