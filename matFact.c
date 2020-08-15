
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

#define RAND01 ((double)random() / (double)RAND_MAX)
#define TRUE 1
#define FALSE 0

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

matrix L, R, R_t, L_aux, R_aux, B;
vector<s_matrix> entries_in_row;
vector<s_matrix> entries_in_column;

void initialize_matrix(matrixPtr M, int lines, int columns);
void random_fill_LR(int nU, int nI, int nF);
void copy(matrix M, matrix M_aux, int lines, int columns);
void transpose(matrix M, matrix M_t, int lines, int columns);
void product_transposed(int nU, int nI, int nF);
void iterate(int numIter, double alpha, int nU, int nI, int nF, s_matrix A);
void print_matrix(matrix M, int lines, int columns);
void print_s_matrix(s_matrix A);
void print_result(s_matrix A, int lines, int columns);
void free_matrix(matrix M, int lines);
int coord_in_A(s_matrix A, int line, int column);

// Allocate memory for MxN matrixes
void initialize_matrix(matrixPtr M, int lines, int columns)
{
	*M = (double **) malloc(lines * sizeof(double *)); 
	for (int i = 0; i < lines; i++) 
		(*M)[i] = (double *) malloc(columns * sizeof(double)); 
}

// As provided in the statement
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
	for (int i = 0; i < lines; i++)
		for (int j = 0; j < columns; j++)
			M_t[j][i] = M[i][j];
}

// Multiply L x R_t
// Using R_t instead of R enhances hit rates in cache
// Values in A are placed as 0
void product_transposed(s_matrix A, int nU, int nI, int nF)
{
	double sum = 0;
	for (int i = 0; i < nU; i++) {
		for (int j = 0; j < nI; j++) {
			for (int k = 0; k < nF; k++) {
				sum += L[i][k] * R_t[j][k];
			}
			B[i][j] = sum;
			sum = 0;
		}
	}
	for (int p = 0; p < A.size(); p++) {
		int i = A[p].i, j = A[p].j;
		B[i][j] = 0;
	}
}

void product_short(s_matrix A, int nU, int nI, int nF)
{
	for (s_entry Aij : A) {
		double sum = 0;
		int i = Aij.i, j = Aij.j;
		for (int k = 0; k < nF; k++) {
			sum += L[i][k] * R_t[j][k];
		}
		B[i][j] = sum;
	}
}

void print_s_matrix(s_matrix A) 
{ 
	for (s_entry Aij : A) {
		cout << Aij.i << " " << Aij.j << " " << Aij.val << endl;
	}
} 

void print_matrix(matrix M, int lines, int columns) 
{ 
	for (int i = 0; i < lines; i++) { 
		for (int j = 0; j < columns; j++)  
			cout << M[i][j] << " ";         
		cout << endl; 
	} 
}

// index matrix A by row: store all values in a single row
s_matrix indexes_of_row(s_matrix A, int line) {
	s_matrix result;
	for (int p = 0; p < A.size(); p++) {
		if (A[p].i == line) 
			result.push_back(A[p]);
		else if (A[p].i > line) // assumption that input file has matrix items ordered by line
			break;
	}
	return result;
}

// index matrix A by column: store all values in a single column
s_matrix indexes_of_column(s_matrix A, int column) {
	s_matrix result;
	for (int p = 0; p < A.size(); p++)
		if (A[p].j == column) 
			result.push_back(A[p]);
	
	return result;
}

// index matrix to simplify inner for cycle in iterate
void index_matrix(s_matrix A, int lines, int columns)
{
	entries_in_row.resize(lines);
	entries_in_column.resize(columns);
	
	for (int i = 0; i < lines; i++) {
	  entries_in_row[i] = indexes_of_row(A, i);
	}

	for (int j = 0; j < columns; j++) {
	  entries_in_column[j] = indexes_of_column(A, j);
	}  
}

// free memory allocated
void free_matrix(matrix M, int lines)
{
	for (int i = 0; i < lines; i++)
		free(M[i]);
	free(M);
}

void iterate(int numIter, double alpha, int nU, int nI, int nF, s_matrix A)
{
	matrix tmp;

	for (int count = 0; count < numIter; count++) {

		product_short(A, nU, nI, nF);

		for (int i = 0; i < nU; i++) {
			for (int k = 0; k < nF; k++) {
				double deltadl = 0;
				for (s_entry Aij : entries_in_row[i]) {
					deltadl += 2 * (Aij.val - B[i][Aij.j]) * (-1) * (R_t[Aij.j][k]);
				}
				L_aux[i][k] = L[i][k] - alpha * deltadl;
			}
		}

		for (int j = 0; j < nI; j++) {
			for (int k = 0; k < nF; k++) {
				double deltadr = 0;
				for (s_entry Aij : entries_in_column[j]) {
					deltadr += 2 * (Aij.val - B[Aij.i][j]) * (-1) * (L[Aij.i][k]);              
				}
				R_aux[j][k] = R_t[j][k] - alpha * deltadr;
			}       
		}
		
		tmp = L;
		L = L_aux;
		L_aux = tmp;

		tmp = R_t;
		R_t = R_aux;
		R_aux = tmp;  
	}
}

void print_result(s_matrix A, int lines, int columns)
{
	for(int i = 0; i < lines; i++) {
		double max = 0;
		int max_j = 0;
		for(int j = 0; j < columns; j++) {
			if (B[i][j] > max) {
				max = B[i][j];
				max_j = j;
			}
		}
		cout << max_j << endl;
	}
}

int main(int argc, char** argv)
{
	string inFile = "";
	string outFile = "";

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
	initialize_matrix(&R_t, nI, nF);
	initialize_matrix(&L_aux, nU, nF);
	initialize_matrix(&R_aux, nI, nF);
	initialize_matrix(&B, nU, nI);

	random_fill_LR(nU, nI, nF);
	index_matrix(A, nU, nI);

	transpose(R, R_t, nF, nI);
	copy(L, L_aux, nU, nF);
	copy(R_t, R_aux, nI, nF);

	/* PROCESSING */

	iterate(numIter, alpha, nU, nI, nF, A);
	product_transposed(A, nU, nI, nF);

	/* RESULT */

	print_result(A, nU, nI);

	/* CLEANUP */

	free_matrix(L, nU);
	free_matrix(R, nF);
	free_matrix(R_t, nI);
	free_matrix(L_aux, nU);
	free_matrix(R_aux, nI);
	free_matrix(B, nU);
}