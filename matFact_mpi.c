// TODO: better way of finding AB_start/AB_t_start?
// idea: each thread can iterate AB and create array at start

#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

#define RAND01 ((double)random() / (double)RAND_MAX)
#define MASTER 0
#define MSG_FROM_MASTER 1
#define MSG_FROM_WORKER 2

typedef struct {
	int i;
	int j;
	double val_A;
	double val_B;
} s_entry;

typedef double** matrix;
typedef matrix* matrixPtr;
typedef vector<s_entry> s_matrix;
MPI_Datatype MPI_ENTRY;

int id, threads;

matrix L, R, R_t, L_aux, R_aux;
s_matrix AB, AB_t;

int L_offset, L_row;
int R_offset, R_row;
int AB_offset, AB_row;
int AB_t_offset, AB_t_row;
int* AB_start; // start of each row in local AB
int* AB_t_start; // start of each row in local AB_t

//only for master
vector<int> L_all_offsets, L_all_rows;
vector<int> R_all_offsets, R_all_rows;
vector<int> AB_all_offsets, AB_all_rows;
vector<int> AB_t_all_offsets, AB_t_all_rows;
vector<int> AB_t_indexes;

void initialize_matrix(matrixPtr M, int lines, int columns);
void free_matrix(matrix M, int lines);
void print_matrix(matrix M, int lines, int columns);
void random_fill_LR(int nU, int nI, int nF);
void copy(matrix M, matrix M_aux, int lines, int columns);
void transpose(matrix M, matrix M_t, int lines, int columns);
bool compare_by_column(s_entry a, s_entry b);
void transpose_B();
void update_B_t();
void initialize(int nU, int nI, int nF);
void calculate_L(double alpha, int nU, int nI, int nF);
void calculate_R(double alpha, int nU, int nI, int nF);
void iterate(int numIter, double alpha, int nU, int nI, int nF);
void free_before_B(int nU, int nI, int nF);
void finalize(int nU, int nI, int nF);
void create_vector_type();


vector<int> sort_indexes(s_matrix &v) {
	vector<int> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	stable_sort(idx.begin(), idx.end(),
		[&v](int i1, int i2) {return v[i1].j < v[i2].j;});
	
	stable_sort(v.begin(), v.end(),
		[&v](s_entry i1, s_entry i2) {return i1.j < i2.j;});

	return idx;
}

void initialize_matrix(matrixPtr M, int lines, int columns) {
	double *data = (double *) malloc(lines * columns * sizeof(double));
	*M = (double **) malloc(lines * sizeof(double *)); 
	for (int i = 0; i < lines; i++) 
		(*M)[i] = &(data[columns * i]); 
}

void free_matrix(matrix M, int lines) {
	free(M[0]);
	free(M);
}

void print_matrix(matrix M, int lines, int columns) { 
	for (int i = 0; i < lines; i++) { 
		for (int j = 0; j < columns; j++)  
			cout << M[i][j] << " ";         
		cout << endl; 
	} 
}

void random_fill_LR(int nU, int nI, int nF) {
	srandom(0);

	for(int i = 0; i < nU; i++)
		for(int j = 0; j < nF; j++)
			L[i][j] = RAND01 / (double) nF;
	
	for(int i = 0; i < nF; i++) 
		for(int j = 0; j < nI; j++)
			R[i][j] = RAND01 / (double) nF;
}

void copy(matrix M, matrix M_aux, int lines, int columns) {
	for (int i = 0; i < lines; i++)
		for (int j = 0; j < columns; j++)
			M_aux[i][j] = M[i][j];
}

void transpose(matrix M, matrix M_t, int lines, int columns) {
	for (int i = 0; i < lines; i++)
		for (int j = 0; j < columns; j++)
			M_t[j][i] = M[i][j];
}

void transpose_B() {
	// cout << "Matrix AB:" << endl;
	// for ( s_entry N : AB) {
	// 	cout << N.i << " " << N.j << " " << N.val_A << " " << N.val_B << endl;
	// }

	AB_t = AB;
	AB_t_indexes = sort_indexes(AB_t);

	// cout << "Matrix AB_t:" << endl;
	// for ( s_entry N : AB_t) {
	// 	cout << N.i << " " << N.j << " " << N.val_A << " " << N.val_B << endl;
	// }

	// cout << "Transposed index: " << endl;
	// for ( int i : AB_t_indexes) {
	// 	cout << AB[i].i << " "<< AB[i].j << " " << endl;
	// }

}

void update_B_t() {
	for (int i = 0; i < AB_t.size(); i++) {
		AB_t[i].val_B = AB[AB_t_indexes[i]].val_B;
	}
}

// calculate all offsets, send to threads, allocate matrices
void initialize(int nU, int nI, int nF) {
	if (id == MASTER) {
		L_all_offsets.resize(threads);
		L_all_rows.resize(threads);
		R_all_offsets.resize(threads);
		R_all_rows.resize(threads);
		AB_all_offsets.resize(threads);
		AB_all_rows.resize(threads);
		AB_t_all_offsets.resize(threads);
		AB_t_all_rows.resize(threads);	

		int L_rows_per_thread = nU / threads;
		int L_extra_rows = nU % threads;
		int offset = 0;

		for (int i = 0; i < threads; i++) {
			L_all_rows[i] = (i < L_extra_rows) ? L_rows_per_thread + 1 : L_rows_per_thread;
			L_all_offsets[i] = offset;
			offset += L_all_rows[i];
		}

		int R_rows_per_thread = nI / threads;
		int R_extra_rows = nI % threads;
		offset = 0;

		for (int i = 0; i < threads; i++) {
			R_all_rows[i] = (i < R_extra_rows) ? R_rows_per_thread + 1 : R_rows_per_thread;
			R_all_offsets[i] = offset;
			offset += R_all_rows[i];
		}

		int p = 0;
		offset = 0;

		for (int i = 0; i < AB.size(); i++) {
			if (AB[i].i >= L_all_rows[p] + L_all_offsets[p]) {
				AB_all_offsets[p] = offset;
				AB_all_rows[p] = i - offset;
				p++;
				offset = i;
			}
		}

		while (p < threads) {
			AB_all_offsets[p] = offset;
			AB_all_rows[p] = AB.size() - offset;
			p++;
			offset = AB.size();
		}

		transpose_B();

		p = 0;
		offset = 0;

		for (int i = 0; i < AB_t.size(); i++) {
			if (AB_t[i].j >= R_all_rows[p] + R_all_offsets[p]) {
				AB_t_all_rows[p] = i - offset;
				AB_t_all_offsets[p] = offset;
				p++;
				offset = i;
			}
		}

		while (p < threads) {
			AB_t_all_offsets[p] = offset;
			AB_t_all_rows[p] = AB_t.size() - offset;
			p++;
			offset = AB_t.size();
		}

		// for (int i = 0; i < threads; i++) {
		// 	cout << "Thread " << i << ": L rows " << L_all_offsets[i] << " to " << L_all_offsets[i] + L_all_rows[i] << endl;
		// 	cout << "Thread " << i << ": R rows " << R_all_offsets[i] << " to " << R_all_offsets[i] + R_all_rows[i] << endl;
		// 	cout << "Thread " << i << ": AB items " << AB_all_offsets[i] << " to " << AB_all_offsets[i] + AB_all_rows[i] << endl;
		// 	cout << "Thread " << i << ": AB_t items " << AB_t_all_offsets[i] << " to " << AB_t_all_offsets[i] + AB_t_all_rows[i] << endl;
		// }

		L_offset = L_all_offsets[0];
		L_row = L_all_rows[0];
		R_offset = R_all_offsets[0];
		R_row = R_all_rows[0];
		AB_offset = AB_all_offsets[0];
		AB_row = AB_all_rows[0];
		AB_t_offset = AB_t_all_offsets[0];
		AB_t_row = AB_t_all_rows[0];

		for (int i = 1; i < threads; i++) {
			MPI_Send(&L_all_offsets[i], 1, MPI_INT, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&L_all_rows[i], 1, MPI_INT, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&R_all_offsets[i], 1, MPI_INT, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&R_all_rows[i], 1, MPI_INT, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&AB_all_offsets[i], 1, MPI_INT, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&AB_all_rows[i], 1, MPI_INT, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&AB_t_all_offsets[i], 1, MPI_INT, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&AB_t_all_rows[i], 1, MPI_INT, i, MSG_FROM_MASTER, MPI_COMM_WORLD);

			// send sections of AB and AB_t
			MPI_Send(&AB[AB_all_offsets[i]], AB_all_rows[i], MPI_ENTRY, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&AB_t[AB_t_all_offsets[i]], AB_t_all_rows[i], MPI_ENTRY, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
		}

		initialize_matrix(&L, nU, nF);
		initialize_matrix(&R, nF, nI);
		initialize_matrix(&R_t, nI, nF);
		initialize_matrix(&L_aux, nU, nF);
		initialize_matrix(&R_aux, nI, nF);

		random_fill_LR(nU, nI, nF);

		transpose(R, R_t, nF, nI);
		copy(L, L_aux, nU, nF);
		copy(R_t, R_aux, nI, nF);
		free_matrix(R, nF);
	}
	else {
		MPI_Status status;

		MPI_Recv(&L_offset, 1, MPI_INT, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&L_row, 1, MPI_INT, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&R_offset, 1, MPI_INT, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&R_row, 1, MPI_INT, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&AB_offset, 1, MPI_INT, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&AB_row, 1, MPI_INT, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&AB_t_offset, 1, MPI_INT, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&AB_t_row, 1, MPI_INT, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);

		AB.resize(AB_row);
		MPI_Recv(&AB[0], AB_row, MPI_ENTRY, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);

		AB_t.resize(AB_t_row);
		MPI_Recv(&AB_t[0], AB_t_row, MPI_ENTRY, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);

		initialize_matrix(&L, nU, nF);
		initialize_matrix(&L_aux, L_row, nF);
		initialize_matrix(&R_t, nI, nF);
		initialize_matrix(&R_aux, R_row, nF);
	}

	AB_start = (int *) malloc(L_row * sizeof(int)); 
	AB_t_start = (int *) malloc(R_row * sizeof(int)); 

	for (int i = 0; i < L_row; i++) {
		int i_off = L_offset + i;
		for (int p = 0; p < AB_row; p++) {
			if (AB[p].i == i_off) {
				AB_start[i] = p;
				break;
			}
		}
	}

	for (int j = 0; j < R_row; j++) {
		int j_off = R_offset + j;
		for (int p = 0; p < AB_t_row; p++) {
			if (AB_t[p].j == j_off) {
				AB_t_start[j] = p;
				break;
			}
		}
	}
}

void calculate_L(double alpha, int nU, int nI, int nF) {
	int workers = threads - 1;

	if (id == MASTER) {
		MPI_Request AB_requests[workers];
		MPI_Request L_requests[workers];
		MPI_Status statuses[workers];

		for (int i = 1; i < threads; i++) {
			MPI_Send(&L[L_all_offsets[i]][0], L_all_rows[i] * nF, MPI_DOUBLE, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&R_t[0][0], nI * nF, MPI_DOUBLE, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
		}

		for (int i = 1; i < threads; i++) {
			MPI_Irecv(&AB[AB_all_offsets[i]], AB_all_rows[i], MPI_ENTRY, i, MSG_FROM_WORKER, MPI_COMM_WORLD, &AB_requests[i-1]);
			MPI_Irecv(&L_aux[L_all_offsets[i]][0], L_all_rows[i] * nF, MPI_DOUBLE, i, MSG_FROM_WORKER, MPI_COMM_WORLD, &L_requests[i-1]);
		}

		for (int p = 0; p < AB_row; p++) {
			double sum = 0;
			s_entry N = AB[p];
			int i = N.i, j = N.j;
			for (int k = 0; k < nF; k++) {
				sum += L[i][k] * R_t[j][k];
			}
			AB[p].val_B = sum;
		}

		for (int i = 0; i < L_row; i++) {
			int i_off = L_offset + i;
			for (int k = 0; k < nF; k++) {
				double deltadl = 0;
				for (int p = AB_start[i]; p < AB_row; p++) {
					s_entry N = AB[p];
					if (N.i == i_off)
						deltadl += 2 * (N.val_A - N.val_B) * (-1) * (R_t[N.j][k]);
					else break;
				}
				L_aux[i][k] = L[i][k] - alpha * deltadl;
			}
		}

		MPI_Waitall(workers, AB_requests, statuses);
		update_B_t();
		MPI_Waitall(workers, L_requests, statuses);
	}
	else {
		MPI_Request request;
		MPI_Status status;
		MPI_Recv(&L[0][0], L_row * nF, MPI_DOUBLE, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&R_t[0][0], nI * nF, MPI_DOUBLE, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);

		for (int p = 0; p < AB_row; p++) {
			double sum = 0;
			s_entry N = AB[p];
			int i = N.i - L_offset, j = N.j;
			for (int k = 0; k < nF; k++) {
				sum += L[i][k] * R_t[j][k];
			}
			AB[p].val_B = sum;
		}

		MPI_Isend(&AB[0], AB_row, MPI_ENTRY, MASTER, MSG_FROM_WORKER, MPI_COMM_WORLD, &request);

		for (int i = 0; i < L_row; i++) {
			int i_off = L_offset + i;
			for (int k = 0; k < nF; k++) {
				double deltadl = 0;
				for (int p = AB_start[i]; p < AB_row; p++) {
					s_entry N = AB[p];
					if (N.i == i_off)
						deltadl += 2 * (N.val_A - N.val_B) * (-1) * (R_t[N.j][k]);
					else break;
				}
				L_aux[i][k] = L[i][k] - alpha * deltadl;
			}
		}

		MPI_Wait(&request, &status);

		MPI_Send(&L_aux[0][0], L_row * nF, MPI_DOUBLE, MASTER, MSG_FROM_WORKER, MPI_COMM_WORLD);

	}
}

void calculate_R(double alpha, int nU, int nI, int nF) {
	int workers = threads - 1;
	if (id == MASTER) {
		MPI_Request requests[workers];
		MPI_Status statuses[workers];

		for (int i = 1; i < threads; i++) {
			MPI_Send(&L[0][0], nU * nF, MPI_DOUBLE, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&R_t[R_all_offsets[i]][0], R_all_rows[i] * nF, MPI_DOUBLE, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&AB_t[AB_t_all_offsets[i]], AB_t_all_rows[i], MPI_ENTRY, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
		}

		for (int i = 1; i < threads; i++) {
			MPI_Irecv(&R_aux[R_all_offsets[i]][0], R_all_rows[i] * nF, MPI_DOUBLE, i, MSG_FROM_WORKER, MPI_COMM_WORLD, &requests[i-1]);
		}

		for (int j = 0; j < R_row; j++) {
			int j_off = R_offset + j;
			for (int k = 0; k < nF; k++) {
				double deltadr = 0;
				for (int p = AB_t_start[j]; p < AB_t_row; p++) {
					s_entry N = AB_t[p];
					if (N.j == j_off)
						deltadr += 2 * (N.val_A - N.val_B) * (-1) * (L[N.i][k]);
					else break;
				}
				R_aux[j][k] = R_t[j][k] - alpha * deltadr;
			}
		}

		MPI_Waitall(workers, requests, statuses);
	}
	else {
		MPI_Status status;
		MPI_Recv(&L[0][0], nU * nF, MPI_DOUBLE, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&R_t[0][0], R_row * nF, MPI_DOUBLE, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&AB_t[0], AB_t_row, MPI_ENTRY, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);

		for (int j = 0; j < R_row; j++) {
			int j_off = R_offset + j;
			for (int k = 0; k < nF; k++) {
				double deltadr = 0;
				for (int p = AB_t_start[j]; p < AB_t_row; p++) {
					s_entry N = AB_t[p];
					if (N.j == j_off)
						deltadr += 2 * (N.val_A - N.val_B) * (-1) * (L[N.i][k]);
					else break;
				}
				R_aux[j][k] = R_t[j][k] - alpha * deltadr;
			}
		}

		MPI_Send(&R_aux[0][0], R_row * nF, MPI_DOUBLE, MASTER, MSG_FROM_WORKER, MPI_COMM_WORLD);
	}
}

void iterate(int numIter, double alpha, int nU, int nI, int nF) {
	matrix tmp;

	for (int count = 0; count < numIter; count++) {

		calculate_L(alpha, nU, nI, nF);

		calculate_R(alpha, nU, nI, nF);

		if (id == MASTER) {
			tmp = L;
			L = L_aux;
			L_aux = tmp;

			tmp = R_t;
			R_t = R_aux;
			R_aux = tmp;  
		}
	} 

}

void print_result(int nU, int nI, int nF) {
	
	free_before_B(nU, nI, nF);
	matrix B;
	initialize_matrix(&B, L_row, nI);
	int workers = threads - 1;

	if (id == MASTER) {
		MPI_Request requests[workers];
		MPI_Status statuses[workers];
		vector<int> result;
		result.resize(nU);

		for (int i = 1; i < threads; i++) {
			MPI_Send(&L[L_all_offsets[i]][0], L_all_rows[i] * nF, MPI_DOUBLE, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&R_t[0][0], nI * nF, MPI_DOUBLE, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&AB[AB_all_offsets[i]], AB_all_rows[i], MPI_ENTRY, i, MSG_FROM_MASTER, MPI_COMM_WORLD);
		}

		for (int i = 1; i < threads; i++)
			MPI_Irecv(&result[L_all_offsets[i]], L_all_rows[i], MPI_INT, i, MSG_FROM_WORKER, MPI_COMM_WORLD, &requests[i-1]);

		// calculate all entries of matrix
		double sum = 0;
		for (int i = 0; i < L_row; i++) {
			for (int j = 0; j < nI; j++) {
				for (int k = 0; k < nF; k++) {
					sum += L[i][k] * R_t[j][k];
				}
				B[i][j] = sum;
				sum = 0;
			}
		}

		// remove entries from A
		for (int p = 0; p < AB_row; p++) {
			s_entry N = AB[p];
			int i = N.i - L_offset, j = N.j;
			B[i][j] = 0;
		}

		// store in vector
		for (int i = 0; i < L_row; i++) {
			double max = 0;
			int max_j = 0;
			for(int j = 0; j < nI; j++) {
				if (B[i][j] > max) {
					max = B[i][j];
					max_j = j;
				}
			}
			result[i] = max_j;
		}

		MPI_Waitall(workers, requests, statuses);

		for (int i = 0; i < nU; i++) {
			cout << result[i] << endl;
		}
	}
	else {
		MPI_Status status;
		vector<int> result;
		result.resize(L_row);

		MPI_Recv(&L[0][0], L_row * nF, MPI_DOUBLE, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&R_t[0][0], nI * nF, MPI_DOUBLE, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&AB[0], AB_row, MPI_ENTRY, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		
		double sum = 0;
		for (int i = 0; i < L_row; i++) {
			for (int j = 0; j < nI; j++) {
				for (int k = 0; k < nF; k++) {
					sum += L[i][k] * R_t[j][k];
				}
				B[i][j] = sum;
				sum = 0;
			}
		}

		for (int p = 0; p < AB_row; p++) {
			s_entry N = AB[p];
			int i = N.i - L_offset, j = N.j;
			B[i][j] = 0;
		}

		for (int i = 0; i < L_row; i++) {
			double max = 0;
			int max_j = 0;
			for(int j = 0; j < nI; j++) {
				if (B[i][j] > max) {
					max = B[i][j];
					max_j = j;
				}
			}
			result[i] = max_j;
		}

		MPI_Send(&result[0], L_row, MPI_INT, MASTER, MSG_FROM_WORKER, MPI_COMM_WORLD);
	}


	free_matrix(B, L_row);
}

void free_before_B(int nU, int nI, int nF) {
	free(AB_start);
	free(AB_t_start);
	AB_t.clear();
	AB_t.shrink_to_fit();

	if (id == MASTER) {
		free_matrix(L_aux, nU);
		free_matrix(R_aux, nI);

		R_all_offsets.clear();
		R_all_offsets.shrink_to_fit();
		R_all_rows.clear();
		R_all_rows.shrink_to_fit();
		AB_t_all_offsets.clear();
		AB_t_all_offsets.shrink_to_fit();
		AB_t_all_rows.clear();
		AB_t_all_rows.shrink_to_fit();
		AB_t_indexes.clear();
		AB_t_indexes.shrink_to_fit();
	}
	else {
		free_matrix(L_aux, L_row);
		free_matrix(R_aux, R_row);
	}
}

void finalize(int nU, int nI, int nF) {
	free_matrix(L, nU);
	free_matrix(R_t, nI);
}

void create_vector_type() {
    const size_t num_members = 4;
    int lengths[num_members] = {1, 1, 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(s_entry, i), offsetof(s_entry, j), offsetof(s_entry, val_A), offsetof(s_entry, val_B)};
    MPI_Datatype types[num_members] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_struct(num_members, lengths, offsets, types, &MPI_ENTRY);
    MPI_Type_commit(&MPI_ENTRY);
}

int main(int argc, char** argv)
{
	int numIter = 0, nF = 0, nU = 0, nI = 0, NNZ = 0;
	int nr = 0, nc = 0;
	double alpha = 0.000, val = 0.000;
	
	string inFile;
	ifstream infile;

	MPI_Init (&argc, &argv);

	MPI_Comm_rank (MPI_COMM_WORLD, &id);
    MPI_Comm_size (MPI_COMM_WORLD, &threads);
	create_vector_type();

	if( argc != 2 ) {
		if (id == MASTER)
			cout << "Usage: ./cfile InputFile\n";
		MPI_Finalize();
		return 1;
	}

	inFile = argv[1];

	infile.open(inFile);
	infile >> numIter;
	infile >> alpha;
	infile >> nF;
	infile >> nU >> nI >> NNZ;

	if(nU < threads || nI < threads) {
		if (id == MASTER)
			cout << "Matrix too small to run with thread number.";
		infile.close();
		MPI_Finalize();
		return 1;
	}

	if (id == MASTER) {
		while(infile >> nr >> nc >> val) {
			s_entry tmp;
			tmp.i = nr;
			tmp.j = nc;
			tmp.val_A = val;
			tmp.val_B = 0;
			AB.push_back(tmp);
		}
	}

	infile.close();

	initialize(nU, nI, nF);

	iterate(numIter, alpha, nU, nI, nF);

	print_result(nU, nI, nF);

	finalize(nU, nI, nF);

	MPI_Finalize();
	return 0;
}