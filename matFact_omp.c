#include <stdlib.h>
#include <omp.h>
#include <bits/stdc++.h>

#define RAND01 ((double)random() / (double)RAND_MAX)
#define TRUE 1
#define FALSE 0

using namespace std;

typedef double** matrix;
typedef matrix* matrixPtr;
typedef vector<int> vi; 
typedef vector<double> vd; 
typedef struct {
    int i;
    int j;
    int val;
} s_entry;

typedef vector<s_entry> s_matrix;

matrix L, R;
vector<s_matrix> entries_in_row;
vector<s_matrix> entries_in_column;

void initialize_matrix(matrixPtr M, int lines, int columns);
void random_fill_LR(int nU, int nI, int nF);
void copy(matrix M, matrix M_aux, int lines, int columns);
void transpose(matrix M, matrix M_t, int lines, int columns);
void product_transposed(matrix R_t, matrix B, int nU, int nI, int nF);
void iterate(int numIter, double alpha, int nU, int nI, int nF, matrix L_aux, matrix R_aux, matrix R_t, s_matrix A, matrix B);
void print_matrix(matrix M, int lines, int columns);
void print_s_matrix(s_matrix A);
void print_result(s_matrix A, matrix B, int lines, int columns);
void free_matrix(matrix M, int lines);
int coord_in_A(s_matrix A, int line, int column);


void initialize_matrix(matrixPtr M, int lines, int columns)
{
    *M = (double **) malloc(lines * sizeof(double *)); 
    for (int i = 0; i < lines; i++) 
        (*M)[i] = (double *) malloc(columns * sizeof(double)); 
}

void random_fill_LR(int nU, int nI, int nF)
{
    srandom(0);

    for(int i = 0; i < nU; i++)
        for(int j = 0; j < nF; j++)
            L[i][j] = RAND01 / (double) nF;
    
    for(int i = 0; i < nF; i++) 
        for(int j = 0; j < nI; j++)
            R[i][j] = RAND01 / (double) nF;

}

void copy(matrix M, matrix M_aux, int lines, int columns)
{
    for (int i = 0; i < lines; i++)
        for (int j = 0; j < columns; j++)
            M_aux[i][j] = M[i][j];
}

void transpose(matrix M, matrix M_t, int lines, int columns)
{
    #pragma omp for collapse(2)
    for (int i = 0; i < lines; i++)
        for (int j = 0; j < columns; j++)
            M_t[j][i] = M[i][j];
}

void product_transposed(matrix R_t, matrix B, int nU, int nI, int nF)
{
    int i, j, k;
    double sum = 0;
    #pragma omp for collapse(2)
    for (i = 0; i < nU; i++) {
        for (j = 0; j < nI; j++) {
            for (k = 0; k < nF; k++) {
                sum += L[i][k] * R_t[j][k];
            }
            B[i][j] = sum;
            sum = 0;
        }
    }
}

void print_s_matrix(s_matrix A) 
{ 
    for_each(A.begin(), A.end(), [](s_entry a) {
        cout << a.i << " " << a.j << " " << a.val << endl;
    });
} 

void print_matrix(matrix M, int lines, int columns) 
{ 
    for (int i = 0; i < lines; i++) { 
        for (int j = 0; j < columns; j++)  
            cout << M[i][j] << " ";         
        cout << endl; 
    } 
}

int coord_in_A(s_matrix A, int line, int column)
{
    for (int p = 0; p < A.size(); p++) {
        if (A[p].i == line && A[p].j == column)
            return TRUE;
        else if ((A[p].i == line && A[p].j > column) || (A[p].i > line)) {
            return FALSE;
        }
    }
    return FALSE;
}

s_matrix indexes_of_row(s_matrix A, int line)
{
    s_matrix result;
    for (int p = 0; p < A.size(); p++) {
        if (A[p].i == line) 
            result.push_back(A[p]);
        else if (A[p].i > line) // assumption that input file has matrix items ordered by line
            break;
    }
    return result;
}

s_matrix indexes_of_column(s_matrix A, int column)
{
    s_matrix result;
    for (int p = 0; p < A.size(); p++)
        if (A[p].j == column) 
            result.push_back(A[p]);
    
    return result;
}

void index_matrix(s_matrix A, int lines, int columns)
{
    entries_in_row.resize(lines);
    entries_in_column.resize(columns);

    for (int p = 0; p < A.size(); p++) {

        if (entries_in_row[A[p].i].empty())
            entries_in_row[A[p].i] = indexes_of_row(A, A[p].i);

        if (entries_in_column[A[p].j].empty())
            entries_in_column[A[p].j] = indexes_of_column(A, A[p].j);

    }
}

void free_matrix(matrix M, int lines)
{
    for (int i = 0; i < lines; i++)
        free(M[i]);
    free(M);
}

void iterate(int numIter, double alpha, int nU, int nI, int nF, matrix L_aux, matrix R_aux, matrix R_t, s_matrix A, matrix B)
{
    matrix tmp;
    #pragma omp parallel if (nU > 100 && nI > 100)
    {
        transpose(R, R_t, nF, nI);
        product_transposed(R_t, B, nU, nI, nF);
    }

    for (int count = 0; count < numIter; count++) {

        #pragma omp parallel if (nU > 20 && nI > 20)
        {
            #pragma omp for schedule (static, 1)
            for (int p = 0; p < A.size(); p++) {
                int i = A[p].i, j = A[p].j;

                for (int k = 0; k < nF; k++) {
                    double deltadl = 0, deltadr = 0;

                    for (s_entry Aij : entries_in_row[i]) {
                        deltadl += 2 * (Aij.val - B[i][Aij.j]) * (-1) * (R[k][Aij.j]);
                    }
                    L_aux[i][k] = L[i][k] - alpha * deltadl;

                    for (s_entry Aij : entries_in_column[j]) {
                        deltadr += 2 * (Aij.val - B[Aij.i][j]) * (-1) * (L[Aij.i][k]);              
                    }
                    R_aux[k][j] = R[k][j] - alpha * deltadr;
                } 
            }

            #pragma omp single
            {
                tmp = L;
                L = L_aux;
                L_aux = tmp;

                tmp = R;
                R = R_aux;
                R_aux = tmp;
            }

            transpose(R, R_t, nF, nI);
            product_transposed(R_t, B, nU, nI, nF);   
        }
    }
}

void print_result(s_matrix A, matrix B, int lines, int columns)
{
    for(int i = 0; i < lines; i++) {
        double max = 0;
        int max_j = 0;
        for(int j = 0; j < columns; j++) {
            if (B[i][j] > max) {
                if (coord_in_A(A, i, j) == FALSE) {
                    max = B[i][j];
                    max_j = j;
                }
            }
        }
        cout << max_j << endl;
    }
}

int main(int argc, char** argv)
{

    string inFile = "";

    int numIter = 0, nF = 0, nU = 0, nI = 0, NNZ = 0;
    int nr = 0, nc = 0;
    double alpha = 0.000, val = 0.000;

    s_matrix A;
    
    ifstream infile;

    if( argc == 2 ) {
      inFile = argv[1];
    }
    else {
      cout << "Usage: ./cfile InputFile\n";
      return 1;
    }

    infile.open(inFile);
    infile >> numIter;
    infile >> alpha;
    infile >> nF;
    infile >> nU >> nI >> NNZ;

    while(infile >> nr >> nc >> val) {
        s_entry tmp;
        tmp.i = nr;
        tmp.j = nc;
        tmp.val = val;
        A.push_back(tmp);
    }     

    infile.close();
    
    /* INITIALIZATION */

    initialize_matrix(&L, nU, nF);
    initialize_matrix(&R, nF, nI);
    random_fill_LR(nU, nI, nF);

    index_matrix(A, nU, nI);

    matrix L_aux, R_aux, R_t, B;
    initialize_matrix(&L_aux, nU, nF);
    initialize_matrix(&R_aux, nF, nI);
    initialize_matrix(&R_t, nI, nF);
    initialize_matrix(&B, nU, nI);

    copy(L, L_aux, nU, nF);
    copy(R, R_aux, nF, nI);

    /* PROCESSING */

    iterate(numIter, alpha, nU, nI, nF, L_aux, R_aux, R_t, A, B);

    /* RESULT */

    print_result(A, B, nU, nI);

    /* CLEANUP */

    free_matrix(L, nU);
    free_matrix(R, nF);
    free_matrix(L_aux, nU);
    free_matrix(R_aux, nF);
    free_matrix(R_t, nI);
    free_matrix(B, nU);
}
