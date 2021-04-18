#include <iostream>
#include <omp.h>
#include <fstream>

using namespace std;

#define BLOCK_SIZE 2

class OpenError {};
class MatrixSizeError {};

void PrintMatrix(double* A, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << A[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

double* ReadMatrixFromFile(const char* file, int& n) {
    // Open file
    ifstream f(file);
    if (f.good() != true) {
        std::cout << "File not found!" << std::endl;
        throw OpenError();
    }

    int r, c;
    // Get matrix size
    f >> r >> c;

    // Matrix should be NxN
    if (r != c) {
        std::cout << "ERROR: row and column not equal" << std::endl;
        throw MatrixSizeError();
    }
    n = r;

    // Matrix allocation
    double *A = new double[n * n * sizeof(double)];

    double elem;

    // Read matrix's elements
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            f >> elem;
            A[i * n + j] = elem;
        }
    }

    return A;
}

void GenerateMatrix(int size, double* A) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            //A[i * size + j] = rand() % 100 + 2;
            A[i * size + j] = 0;
        }
    }
}

void DiagonalMatrixDecomposition(int offset, int size, int &N, double* A, double* L, double* U) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            U[N * (offset + i) + offset + j] = A[N * (offset + i) + offset + j];
        }
    }

    for (int i = 0; i < size; i++) {
        L[N * (offset + i) + offset + i] = 1;

        for (int k = i + 1; k < size; k++) {
            double mu = U[N * (offset + k) + offset + i] / U[N * (offset + i) + offset + i];

            for (int j = i; j < size; j++) {
                U[N * (offset + k) + offset + j] -= mu * U[N * (offset + i) + offset + j];
            }


            L[N * (offset + k) + offset + i] = mu;
            L[N * (offset + i) + offset + k] = 0;
        }
    }

    for (int i = 1; i < size; i++) {
        for (int j = 0; j < i; j++) {
            U[N * (offset + i) + offset + j] = 0;
        }
    }
}

void SolveUpper(int offset, int size, int& N, double* A, double* L, double* U) {
    int row_num;
    int col_num;

    for (int k = 0; k < N - offset - BLOCK_SIZE; k++) {
        row_num = offset * N;
        col_num = offset + BLOCK_SIZE;

        U[row_num + col_num + k] = A[row_num + col_num + k];

        for (int i = 1; i < size; i++) {
            U[row_num + i * N + col_num + k] = A[row_num + i * N + col_num + k];
            std::cout << "U" << offset + i << col_num + k << " = " << "A" << offset + i << col_num + k << std::endl;

            for (int j = 0; j < i; j++) {
                U[row_num + i * N + col_num + k] -= L[row_num + i * N + j] * U[row_num + j * N + col_num + k];
                
                std::cout << "U" << row_num + i * N + col_num + k << " (" << U[row_num + i * N + col_num + k] << ") -=" \
                    << "L" << row_num + i * N + j << " (" << L[row_num + i * N + j] << ") * " \
                    << "U" << row_num + j * N + col_num + k << " (" << U[row_num + j * N + col_num + k] << ")" << std::endl;
            }
        }
    }
}

void SolveLower(int offset, int size, int& N, double* A, double* L, double* U) {

}

void LU_Decomposition(double* A, double* L, double* U, int n) {
    for (int offset = 0; offset < n; offset += BLOCK_SIZE) {
        if (n - offset <= BLOCK_SIZE) {
            DiagonalMatrixDecomposition(offset, n - offset, n, &(*A), &(*L), &(*U));
            //PrintMatrix(&(*U), n);
            //std::cout << "DIAG IN IF" << std::endl;
            break;
        }
        DiagonalMatrixDecomposition(offset, BLOCK_SIZE, n, &(*A), &(*L), &(*U));
        //PrintMatrix(&(*U), n);
        //std::cout << "DIAG OUT IF" << std::endl;

        SolveUpper(offset, BLOCK_SIZE, n, &(*A), &(*L), &(*U));
        //PrintMatrix(&(*U), n);
        //std::cout << "DIAG AFTER UPPER" << std::endl;
        //SolveLower(offset, BLOCK_SIZE, n, &(*A), &(*L), &(*U));
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "ERROR: incorrect arguments count" << std::endl;
        return -1;
    }

    int n = 6;
    double* A;
    const char *file_path = argv[1];

    double *L = new double[n * n * sizeof(double)];
    double *U = new double[n * n * sizeof(double)];

    A = ReadMatrixFromFile(file_path, n);
    GenerateMatrix(6, &(*L));
    GenerateMatrix(6, &(*U));

    std::cout << "Matrix A:" << std::endl;
    PrintMatrix(&(*A), n);
    std::cout << "Matrix L:" << std::endl;
    PrintMatrix(&(*L), n);
    std::cout << "Matrix U:" << std::endl;
    PrintMatrix(&(*U), n);

    std::cout << "Calculation..." << std::endl;
    LU_Decomposition(&(*A), &(*L), &(*U), n);
    std::cout << "Calculation...DONE" << std::endl;

    std::cout << "Matrix A:" << std::endl;
    PrintMatrix(&(*A), n);
    std::cout << "Matrix L:" << std::endl;
    PrintMatrix(&(*L), n);
    std::cout << "Matrix U:" << std::endl;
    PrintMatrix(&(*U), n);

    return 0;
}