#include <iostream>
#include <vector>
#include "omp.h"
#include <fstream>
#include <chrono>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace std::chrono;

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

    double eps = 1e-10;

    // Boundary
#pragma omp parallel for
    for (int i = 0; i <= task.n; i++) {
        v[i * (task.m + 1)] = task.bottom_condition(i * h);
        v[i * (task.m + 1) + task.m] = task.top_condition(i * h);
    }
#pragma omp parallel for
    for (int j = 0; j <= task.m; j++) {
        v[j] = task.left_condition(j * k);
        v[task.n * (task.m + 1) + j] = task.right_condition(j * k);
    }

    // The function of external impact
    vector<vector<double>> f(task.n + 1, vector<double>(task.m + 1, 0.0));
#pragma omp parallel for
    for (int i = 0; i <= task.n; i++) {
        for (int j = 0; j <= task.m; j++) {
            f[i][j] = -task.f(i * h, j * k);
        }
    }

    int j;
    int NT;
    double tmp;
    double err;
    double prev;

    int top_index;
    int bot_index;
    int left_index;
    int right_index;
    int curr_index;


#pragma omp parallel sections
    {
        NT = omp_get_num_threads();
    }

    double* errs = new double[NT];


    do {
        err = 0.;

#pragma omp parallel for
        for (int i = 0; i < NT; ++i) {
            errs[i] = 0.;
        }

        for (int k = 0; k < task.n + task.m - 3; ++k) {
            int start = min(1 + k, task.n - 1);
            int finish = max(1, k - task.m + 3);

#pragma omp parallel for private(j, tmp, prev, curr_index, top_index, bot_index, left_index, right_index)
            for (int i = start; i >= finish; --i) {

                int tid = omp_get_thread_num();

                j = task.m - (k - i + 2);

                curr_index = i * (task.m + 1) + j;

                top_index = curr_index + 1;
                bot_index = curr_index - 1;
                left_index = curr_index - (task.m + 1);
                right_index = curr_index + (task.m + 1);

                prev = v[curr_index];

                v[curr_index] = -D * prev;
                v[curr_index] += h_ * v[left_index];
                v[curr_index] += h_ * v[right_index];
                v[curr_index] += k_ * v[bot_index];
                v[curr_index] += k_ * v[top_index];
                v[curr_index] += f[i][j];
                v[curr_index] *= omega;
                v[curr_index] += D * prev;
                v[curr_index] /= D;

                tmp = fabs(v[curr_index] - prev);

                if (tmp > errs[tid]) {
                    errs[tid] = tmp;
                }

                //if (tmp > err) err = tmp;
            }
        }

        for (int i = 0; i < NT; ++i) {
            if (errs[i] > err)
                err = errs[i];
        }

    } while (err > eps);
    delete[] errs;
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

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    heat_dirichlet_sor(t, v);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<double, std::milli> time_span = t2 - t1;

    std::cout << "Duration time: " << time_span.count() << " milliseconds.";
    
    PrintArray((t.n + 1) * (t.m + 1), v);
    
    delete[] v;
    return 0;
}