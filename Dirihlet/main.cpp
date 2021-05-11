#include <iostream>
#include <vector>
#include "omp.h"
#include "Processing.h"
#include <fstream>
#include <algorithm>
#include <chrono>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace std::chrono;

void update_vector(heat_task &task, double* v, double &h, double &k) {

    /*
        Calculating values at the boundary
    */
#pragma omp parallel for
    for (int i = 0; i <= task.n; ++i) {
        v[i * (task.m + 1)] = task.bottom_condition(i * h);
        v[i * (task.m + 1) + task.m] = task.top_condition(i * h);
    }
#pragma omp parallel for
    for (int j = 0; j <= task.m; ++j) {
        v[j] = task.left_condition(j * k);
        v[task.n * (task.m + 1) + j] = task.right_condition(j * k);
    }

    /*
        Nullifying the remaining elements of the vector
    */

#pragma omp parallel for
    for (int i = 1; i < task.n; ++i) {
        for (int j = 1; j < task.m; j++)
            v[i * (task.m + 1) + j] = 0.;
    }
}

void external_impact(vector<vector<double>>& f, heat_task& task, double& h, double& k) {
    /*
        Function of external impact
    */
#pragma omp parallel for
    for (int i = 0; i <= task.n; i++) {
        for (int j = 0; j <= task.m; j++) {
            f[i][j] = -task.f(i * h, j * k);
        }
    }
}

void heat_dirichlet_sor(heat_task task, double* v) {
    double h = task.X / task.n;
    double k = task.Y / task.m;

    double h_ = 1 / pow(h, 2);
    double k_ = 1 / pow(k, 2);

    double omega = 2 / (1 + sin(M_PI * min(h, k) / 2));
    double D = 2 * (h_ + k_);

    update_vector(task, v, h, k);

    vector<vector<double>> f(task.n + 1, vector<double>(task.m + 1, 0.0));

    external_impact(f, task, h, k);

    vector<vector<double>> nu(task.n + 1, vector<double>(task.m + 1, 0.0));

    double prev;

    int top_index;
    int bot_index;
    int left_index;
    int right_index;
    int curr_index;


    int max_iter = 1.5 / min(h, k) / M_PI * log(1 / 1e-7);

    for (int iteration = 0; iteration < max_iter; iteration++) {
        for (int k = 0; k < task.n + task.m - 3; ++k) {

            int start = min(1 + k, task.n - 1);
            int finish = max(1, k - task.m + 3);

#pragma omp parallel for private(prev, curr_index, top_index, bot_index, left_index, right_index)
            for (int i = start; i >= finish; --i) {

                int j = task.m - (k - i + 2);

                curr_index = i * (task.m + 1) + j;

                top_index = curr_index + 1;
                bot_index = curr_index - 1;
                left_index = curr_index - (task.m + 1);
                right_index = curr_index + (task.m + 1);

                prev = v[curr_index];

                v[curr_index] = omega * (h_ * (v[left_index] + v[right_index]) + k_ * (v[top_index] + v[bot_index]) + f[i][j]) + \
                    (1 - omega) * D * prev;
                v[curr_index] /= D;

            }
        }
    }

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