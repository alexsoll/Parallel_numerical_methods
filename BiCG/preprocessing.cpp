#include "preprocessing.h"

void ReadMatrixFromFile(const char* file, CRSMatrix& matrix, double* b) {
    ifstream f(file);
    if (f.good() != true) {
        std::cout << "File not found!" << std::endl;
        throw OpenError();
    }

    int r, c;
    f >> r >> c;

    matrix.n = r;
    matrix.m = c;

    std::string s;

    matrix.val = vector<double>();
    matrix.colIndex = vector<int>();
    matrix.rowPtr = vector<int>();

    double elem;

    f >> s;
    if (s.find("nz") != std::string::npos) {
        f >> matrix.nz;
        for (int i = 0; i < matrix.nz; i++) {
            f >> elem;
            matrix.val.push_back(elem);
        }
        for (int i = 0; i < matrix.nz; i++) {
            f >> elem;
            matrix.colIndex.push_back(elem);
        }
        for (int i = 0; i <= matrix.n; i++) {
            f >> elem;
            matrix.rowPtr.push_back(elem);
        }
        for (int i = 0; i < matrix.n; i++) {
            f >> elem;
            b[i] = elem;
        }
    }
    else {
        throw DataError();
    }
}
