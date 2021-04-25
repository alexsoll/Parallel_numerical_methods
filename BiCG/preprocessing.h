#pragma once
#include <iostream>
#include <vector>
#include "omp.h"
#include <fstream>

using namespace std;

class OpenError {};
class MatrixSizeError {};
class DataError {};

struct CRSMatrix
{
    int n;
    int m;
    int nz;
    vector<double> val;
    vector<int> colIndex;
    vector<int> rowPtr;
};

void ReadMatrixFromFile(const char* file, CRSMatrix& matrix, double* b);