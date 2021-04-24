#include <iostream>
#include <vector>
#include "omp.h"
#include <fstream>

using namespace std;

struct CRSMatrix
{
    int n;
    int m;
    int nz;
    vector<double> val;
    vector<int> colIndex;
    vector<int> rowPtr;
};

void PrintVector(int size, double* a) {
    std::cout << "( ";
    for (int i = 0; i < size; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << ")" << std::endl;
}

void GenerateStartSolution(int& n, double* x) {
    for (int i = 0; i < n; i++) {
        x[i] = 1;
    }
}


void FindR(CRSMatrix& A, double* x0, double* b, double* r) {
    for (int i = 0; i < A.n; i++) {
        double sum = 0;
        int c = A.rowPtr[i + 1] - A.rowPtr[i];

        for (int j = 0; j < c; j++) {
            sum += A.val[A.rowPtr[i] + j] * x0[A.colIndex[A.rowPtr[i] + j]];
        }

        r[i] = b[i] - sum;
    }
}

void SLE_Solver_CRS_BICG(CRSMatrix& A, double* b, double eps, int max_iter, double* x, int& count) {
    int n = A.n;

    double* r = new double[n * sizeof(double)];
    double* r_ = new double[n * sizeof(double)];

    double* p = new double[n * sizeof(double)];
    double* p_ = new double[n * sizeof(double)];

    double alpha, beta;

    GenerateStartSolution(n, x);
    FindR(A, x, b, r);

    PrintVector(A.n, r);


}

int main(int argc, char* argv[]) {
    /*if (argc < 2) {
        std::cout << "ERROR: incorrect arguments count" << std::endl;
        return -1;
    }*/

    vector<double> v = { 10, -2, 3, 9, 3, 7, 8, 7, 3, 8, 7, 5, 8, 9, 9, 13, 4, 2, -1};
    vector<int> colIndex = { 0, 3, 0, 1, 5, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4, 5, 1, 4, 5};
    vector<int> rowPtr = { 0, 2, 5, 8, 12, 16, 19};

    CRSMatrix matrix = {};
    matrix.n = 6;
    matrix.m = 6;
    matrix.val = v;
    matrix.colIndex = colIndex;
    matrix.rowPtr = rowPtr;
    matrix.nz = matrix.val.size();

    double b[] = {1, 2, 3, 2, 1, 5};
    double *x = new double[matrix.n * sizeof(double)];


    int count = 0;
    SLE_Solver_CRS_BICG(matrix, b, 0.001, 100, x, count);


    return 0;
}