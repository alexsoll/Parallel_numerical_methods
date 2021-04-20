#include <iostream>
#include <omp.h>
#include <fstream>

using namespace std;

#define BLOCK_SIZE 32

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

    #pragma omp parallel for private(row_num, col_num)
    for (int k = 0; k < N - offset - BLOCK_SIZE; k++) {
        row_num = offset * N;
        col_num = offset + BLOCK_SIZE;

        U[row_num + col_num + k] = A[row_num + col_num + k];

        for (int i = 1; i < size; i++) {
            U[row_num + i * N + col_num + k] = A[row_num + i * N + col_num + k];

            for (int j = 0; j < i; j++) {
                U[row_num + i * N + col_num + k] -= L[row_num + i * N + j + offset] * U[row_num + j * N + col_num + k];
            }
        }
    }
}

void SolveLower(int offset, int size, int& N, double* A, double* L, double* U) {
    int row_num;
    int col_num;

    #pragma omp parallel for private(row_num, col_num)
    for (int k = 0; k < N - offset - BLOCK_SIZE; k++) {
        row_num = (offset + BLOCK_SIZE) * N;
        col_num = offset;

        L[row_num + k * N + col_num] = A[row_num + k * N + col_num] / U[offset * N + col_num];

        for (int i = 1; i < size; i++) {
            L[row_num + k * N + col_num + i] = A[row_num + k * N + col_num + i]; // U[offset * N + col_num];

            for (int j = 0; j < i; j++) {
                L[row_num + k * N + col_num + i] -= L[row_num + k * N + j + offset] * U[(offset + j) * N + col_num + i];
            }

            L[row_num + k * N + col_num + i] /= U[(offset + i) * N  + col_num + i];
        }
    }
}


void UpdateDiagonalSubmatrix(int offset, int size, int& N, double* A, double* L, double* U) {

    #pragma omp parallel for if (N - offset > 1000)
    for (int i = 0; i < N - offset - BLOCK_SIZE; i++) {
        for (int j = 0; j < N - offset - BLOCK_SIZE; j++) {
            double sum = 0;

            for (int k = 0; k < BLOCK_SIZE; k++) {
                sum += L[(offset + BLOCK_SIZE + i) * N + k + offset] * U[(offset + k) * N + offset + BLOCK_SIZE + j];
            }

            A[(offset + BLOCK_SIZE + i) * N + offset + BLOCK_SIZE + j] -= sum;
        }
    }
}


void LU_Decomposition(double* A, double* L, double* U, int n) {
    for (int offset = 0; offset < n; offset += BLOCK_SIZE) {
        if (n - offset <= BLOCK_SIZE) {
            DiagonalMatrixDecomposition(offset, n - offset, n, &(*A), &(*L), &(*U));
            break;
        }

        DiagonalMatrixDecomposition(offset, BLOCK_SIZE, n, &(*A), &(*L), &(*U));

        #pragma omp parallel sections 
        {
            #pragma omp section 
            {
                SolveUpper(offset, BLOCK_SIZE, n, &(*A), &(*L), &(*U));
            }
            #pragma omp section 
            {
                SolveLower(offset, BLOCK_SIZE, n, &(*A), &(*L), &(*U));
            }
        }
        UpdateDiagonalSubmatrix(offset, BLOCK_SIZE, n, &(*A), &(*L), &(*U));

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

    LU_Decomposition(&(*A), &(*L), &(*U), n);
    PrintMatrix(&(*L), n);
    PrintMatrix(&(*U), n);

    return 0;
}