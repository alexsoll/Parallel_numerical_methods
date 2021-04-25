#include <iostream>
#include <vector>
#include "omp.h"
#include "preprocessing.h"
#include <fstream>

using namespace std;

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

void GetR0(CRSMatrix& A, double* x0, double* b, double* r) {
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

void Multiplication(CRSMatrix& A, double* v, double* res) {
#pragma omp parallel for
    for (int i = 0; i < A.n; i++) {
        res[i] = 0.0;
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; j++) {
            res[i] += A.val[j] * v[A.colIndex[j]];
        }
    }
}

double Scalar(double* a, double* b, int & size) {
    double res = 0;
#pragma omp parallel for reduction(+: res)
    for (int i = 0; i < size; i++) {
        res += a[i] * b[i];
    }
    return res;
}

void GetAlpha(double* r, double* r_, CRSMatrix& A, double* p, double* p_, double &alpha) {
    double* tmp = new double[A.n * sizeof(double)];

    Multiplication(A, p, tmp);

    alpha = Scalar(r, r_, A.n) / Scalar(tmp, p_, A.n);
}

void GetBeta(double* r, double* r_, double* r_prev, double* r_prev_, double& beta, int size) {
    beta = Scalar(r, r_, size) / Scalar(r_prev, r_prev_, size);
}

void Addition(double* l, double* r, double* res, double coef, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        res[i] = l[i] + coef * r[i];
    }
}

double Norma(double* a, int size) {
    double sum = 0.0;
#pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < size; i++) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

void SLE_Solver_CRS_BICG(CRSMatrix& A, double* b, double eps, int max_iter, double* x, int& count) {
    int n = A.n;

    double* r = new double[n * sizeof(double)];
    double* r_prev = new double[n * sizeof(double)];
    double* r_ = new double[n * sizeof(double)];
    double* r_prev_ = new double[n * sizeof(double)];

    double* p = new double[n * sizeof(double)];
    double* p_ = new double[n * sizeof(double)];

    double* tmp = new double[n * sizeof(double)];

    double alpha, beta;

    GenerateStartSolution(n, x);
    GetR0(A, x, b, r);

    CopyArray(r, r_, n);
    CopyArray(r, p, n);
    CopyArray(r, p_, n);

    CRSMatrix AT = Transponse(A);
    

    while (true) {
        count++;

        GetAlpha(r, r_, A, p, p_, alpha);
        Addition(x, p, x, alpha, n);

        Multiplication(A, p, tmp); 
        CopyArray(r, r_prev, n);
        Addition(r, tmp, r, -alpha, A.n);

        Multiplication(AT, p_, tmp);
        CopyArray(r_, r_prev_, n);
        Addition(r_, tmp, r_, -alpha, n);

        GetBeta(r, r_, r_prev, r_prev_, beta, n);

        if (abs(beta) < eps) {
            std::cout << "betta < eps" << std::endl;
            std::cout << "Result: x = ";
            PrintArray(n, x);
            std::cout << "Count: " << count << std::endl;
            break;
        }
        if (Norma(r, n) < eps) {
            std::cout << "||r|| < eps" << std::endl;
            std::cout << "Result: x = ";
            PrintArray(n, x);
            std::cout << "Count: " << count << std::endl;
            break;
        }
        if (count == max_iter) {
            std::cout << "Exceeded the number of allowed iterations" << std::endl;
            std::cout << "Result: x = ";
            PrintArray(n, x);
            break;
        }

        Addition(r, p, p, beta, n);
        Addition(r_, p_, p_, beta, n);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "ERROR: incorrect arguments count" << std::endl;
        return -1;
    }

    const char* file_path = argv[1];

    int n = 3;

    CRSMatrix matrix;
    double* b = new double[n * sizeof(double)];
    double* x = new double[n * sizeof(double)];

    ReadMatrixFromFile(file_path, matrix, b);

    int count = 0;
    SLE_Solver_CRS_BICG(matrix, b, 0.001, 100, &(*x), count);

    return 0;
}