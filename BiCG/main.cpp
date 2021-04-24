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

template <typename T>
void PrintVector(vector<T> a) {
    std::cout << "( ";
    for (int i = 0; i < a.size(); i++) {
        std::cout << a[i] << " ";
    }
    std::cout << ")" << std::endl;
}

void PrintArray(int size, double *a) {
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

void CopyArray(double* source, double* destination, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        destination[i] = source[i];
    }
}


void FindR0(CRSMatrix& A, double* x0, double* b, double* r) {
#pragma omp parallel for
    for (int i = 0; i < A.n; i++) {
        double sum = 0;
        int c = A.rowPtr[i + 1] - A.rowPtr[i];

        for (int j = 0; j < c; j++) {
            sum += A.val[A.rowPtr[i] + j] * x0[A.colIndex[A.rowPtr[i] + j]];
        }

        r[i] = b[i] - sum;
    }
}

CRSMatrix Transponse(CRSMatrix& A) {
    CRSMatrix AT = {};
    AT.n = A.m;
    AT.m = A.n;
    AT.nz = A.nz;

    AT.rowPtr = vector<int>(AT.n + 1);
    AT.colIndex = vector<int>(AT.nz);
    AT.val = vector<double>(AT.nz);

    for (int i = 0; i < A.nz; i++) {
        AT.rowPtr[A.colIndex[i] + 1]++;
    }

    int S = 0;
    int tmp;

    for (int i = 1; i <= A.n; i++) {
        tmp = AT.rowPtr[i];
        AT.rowPtr[i] = S;
        S = S + tmp;
    }

    int RIndex, IIndex, c, col, j1, j2;
    double v;

    for (int i = 0; i < A.n; i++) {
        c = A.rowPtr[i + 1] - A.rowPtr[i];
        col = i;
        j1 = A.rowPtr[i];
        j2 = A.rowPtr[i + 1];

        for (int j = j1; j < j2; j++) {
            v = A.val[j];
            RIndex = A.colIndex[j];
            IIndex = AT.rowPtr[RIndex + 1];
            AT.val[IIndex] = v;
            AT.colIndex[IIndex] = col;
            AT.rowPtr[RIndex + 1]++;
        }
    }
    return AT;
}

void SLE_Solver_CRS_BICG(CRSMatrix& A, double* b, double eps, int max_iter, double* x, int& count) {
    int n = A.n;

    double* r = new double[n * sizeof(double)];
    double* r_ = new double[n * sizeof(double)];

    double* p = new double[n * sizeof(double)];
    double* p_ = new double[n * sizeof(double)];

    double alpha, beta;

    GenerateStartSolution(n, x);
    FindR0(A, x, b, r);

    CopyArray(r, r_, n);
    CopyArray(r, p, n);
    CopyArray(r, p_, n);
    
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


    double b[] = {1, 1, 1, 1, 1, 1};

    /*vector<double> v = { 3, 7, 8, 9, 15, 16};
    vector<int> colIndex = { 1, 3, 2, 0, 2, 3};
    vector<int> rowPtr = { 0, 2, 3, 3, 6};
    CRSMatrix matrix = {};
    matrix.n = 4;
    matrix.m = 4;
    matrix.val = v;
    matrix.colIndex = colIndex;
    matrix.rowPtr = rowPtr;
    matrix.nz = matrix.val.size();
    double b[] = {2, -2, 2};*/

    double* x = new double[matrix.n * sizeof(double)];

    int count = 0;
    SLE_Solver_CRS_BICG(matrix, b, 0.001, 100, x, count);

    return 0;
}