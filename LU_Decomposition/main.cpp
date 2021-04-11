#include <iostream>
#include <omp.h>
#include <fstream>

using namespace std;

class OpenError {};
class MatrixSizeError {};

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
    std::cout << "r: " << r << std::endl;
    std::cout << "c: " << c << std::endl;

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
            std::cout << elem << " ";
            A[i * n + j] = elem;
        }
        std::cout << std::endl;
    }

    return A;
}

void LU_Decomposition(double* A, double* L, double* U, int n);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "ERROR: incorrect arguments count" << std::endl;
        return -1;
    }

    int n;
    double* A;
    const char *file_path = argv[1];

    A = ReadMatrixFromFile(file_path, n);

    return 0;
}