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
	float finalmatrixA[matrixSize/(int)sqrt(size)][matrixSize/(int)sqrt(size)]; // Second scatter buffer which contains the matrix partition for the processor
	float allrowsA[matrixSize/(int)sqrt(size)][matrixSize];
	float allcolumnsA[matrixSize][matrixSize/(int)sqrt(size)];
	float localmatrixB[matrixSize/(int)sqrt(size)][matrixSize]; // First scatter buffer that holds all the rows a row of processors will need
	float finalmatrixB[matrixSize/(int)sqrt(size)][matrixSize/(int)sqrt(size)]; // Second scatter buffer which contains the matrix partition for the processor
	float allrowsB[matrixSize/(int)sqrt(size)][matrixSize];
	float allcolumnsB[matrixSize][matrixSize/(int)sqrt(size)];
	float localresult[matrixSize/(int)sqrt(size)][matrixSize/(int)sqrt(size)];
	float rowresult[matrixSize/(int)sqrt(size)][matrixSize];
	float globalResult[matrixSize][matrixSize];
	float testResult[matrixSize][matrixSize];

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
		
		
		printf("Matrix multiplication result is\n");
		for(int i = 0; i < matrixSize; i++) {
			for(int y = 0; y < matrixSize; y++) {
				globalResult[i][y] = 0;
				for(int z = 0; z < matrixSize; z++) {
					globalResult[i][y] += (globalMatrixA[i][z] * globalMatrixB[z][y]);
				}
				printf(" %lf ", globalResult[i][y]);
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

	/* All of the row pieces scattered among the processors are gathered up on each processor in a row group. This is done for matrix A as the multiplication will be done
 	 * by multiplying the rows of A with the columns of B */	
	for(int i = 0; i < matrixSize/(int)sqrt(size); i++){
		MPI_Allgather(finalmatrixA[i], (matrixSize/(int)sqrt(size)), MPI_FLOAT, &allrowsA[i], (matrixSize/(int)sqrt(size)), MPI_FLOAT, row_comm); 
	}
	
	/* Gather up all the columns pieces in a column on each processor in that column */
	MPI_Allgather(finalmatrixB, (matrixSize/(int)sqrt(size))*(matrixSize/(int)sqrt(size)), MPI_FLOAT, &allcolumnsB, (matrixSize/(int)sqrt(size))*(matrixSize/(int)sqrt(size)), MPI_FLOAT, column_comm);

	/* Compute the values of the matrix multiplication locally on the processor */
	for(int i = 0; i < matrixSize/(int)sqrt(size); i++) {
		for(int y = 0; y < matrixSize/(int)sqrt(size); y++) {
			localresult[i][y] = 0;
			for(int z = 0; z < matrixSize; z++) {
				localresult[i][y] += (allrowsA[i][z] * allcolumnsB[z][y]);
			}
		}
	}

	/* With the local result calculated reverse the scatter process and gather up the results */

	for(int i = 0; i < matrixSize/(int)sqrt(size); i++) {	
		MPI_Allgather(localresult[i], matrixSize/(int)sqrt(size), MPI_FLOAT, &rowresult[i], matrixSize/(int)sqrt(size), MPI_FLOAT, row_comm);
	}
	
	/* At this point each row group processor contains all the multiplication results for that row partition. Gather up these results on the root process using the column communicator */
	MPI_Gather(rowresult, (matrixSize*matrixSize)/(int)sqrt(size), MPI_FLOAT, &testResult, (matrixSize*matrixSize)/(int)sqrt(size), MPI_FLOAT, 0, column_comm);
	
	/* Print out the multiplication result on the world root process just to ensure the results match */
	if(rank == 0) {
		printf("Matrix multiplication result is\n");
		for(int i = 0; i < matrixSize; i++) {
			for(int y = 0; y < matrixSize; y++) {
				printf(" %lf ", testResult[i][y]);
			}
			printf("\n");
		}
	}
	MPI_Finalize();
}
