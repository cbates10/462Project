#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){
	int matrixSize = 16;
	MPI_Init(NULL, NULL);
	int rank;
	int size;
	float globalMatrixA[matrixSize][matrixSize];
	float globalMatrixB[matrixSize][matrixSize];
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int baseCommunicatorRanks[(int) sqrt(size)];
	float localmatrixA[matrixSize/(int)sqrt(size)][matrixSize]; // First scatter buffer that holds all the rows a row of processors will need
	float finalmatrixA[matrixSize/(int)sqrt(size)][matrixSize/(int)sqrt(size)]; // Send scatter buffer which contains the matrix partition for the processor
	float localmatrixB[matrixSize/(int)sqrt(size)][matrixSize]; // First scatter buffer that holds all the rows a row of processors will need
	float finalmatrixB[matrixSize/(int)sqrt(size)][matrixSize/(int)sqrt(size)]; // Send scatter buffer which contains the matrix partition for the processor
	double t1, t2;
	
	MPI_Comm row_comm;
	MPI_Comm column_comm;
	int rowColor = rank/(int)sqrt(size);
	int columnColor = rank%(int)sqrt(size);
	/* Custom communicators that will make scattering and gathering matrix information easier. Each processor forms a communicator with the processors
 	 * in the same row and same column */
	MPI_Comm_split(MPI_COMM_WORLD, rowColor, rank, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, columnColor, rank, &column_comm);

		
	/* Initialize the matrices with "random" numbers */	
	if(rank == 0){
		int subMatrixDimension = matrixSize / size;
		int subMatrixSize = subMatrixDimension * subMatrixDimension;
		int columnIndex;
		t1 = MPI_Wtime();
		srand(0);
		for(int i = 0; i < matrixSize; i++){
			printf("Matrix A contents are :\n");
			for(int y = 0; y < matrixSize; y++){
				globalMatrixA[i][y] = ((float)rand() / (float)RAND_MAX) * 2 - 1;
				printf(" %lf ", globalMatrixA[i][y]);
				globalMatrixB[i][y] = ((float)rand() / (float)RAND_MAX) * 2 - 1;
			}
			printf("\n");
		}
	}
	/* This first scatter breaks apart the rows of the globalmatrix and passes the rows that are mapped to a processor to the first processor in the row  */
	MPI_Scatter(globalMatrixA, (matrixSize*matrixSize)/(int)sqrt(size), MPI_FLOAT, &localmatrixA, (matrixSize*matrixSize)/(int)sqrt(size), MPI_FLOAT, 0, column_comm);
	MPI_Scatter(globalMatrixB, (matrixSize*matrixSize)/(int)sqrt(size), MPI_FLOAT, &localmatrixB, (matrixSize*matrixSize)/(int)sqrt(size), MPI_FLOAT, 0, column_comm);

	/* With the first processor in each processor row now containing all the rows for that processor row, the rows are scattered among all the processors in the row */
	for(int i = 0; i < matrixSize/(int)sqrt(size); i++) {
		MPI_Scatter(localmatrixA[i], matrixSize/(int)sqrt(size), MPI_FLOAT, &finalmatrixA[i], matrixSize/(int)sqrt(size), MPI_FLOAT, 0, row_comm);
		MPI_Scatter(localmatrixB[i], matrixSize/(int)sqrt(size), MPI_FLOAT, &finalmatrixB[i], matrixSize/(int)sqrt(size), MPI_FLOAT, 0, row_comm);
	}

	if(rank == 4) {
		printf("%d is the rank\n", rank);
		for(int i = 0; i < matrixSize/(int)sqrt(size); i++) {
			for(int y = 0; y < matrixSize/(int)sqrt(size); y++) {
				printf(" %lf ", finalmatrix[i][y]);
			}
		printf("\n");
		}
	}
	MPI_Finalize();
}
