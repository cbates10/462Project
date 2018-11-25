#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int mod(int a, int b) {
	return ((((a)%(b))+(b))%(b));
}

int main(){
	int matrixSize = 256;
	MPI_Init(NULL, NULL);
	int rank;
	int rowRank;
	int colRank;
	int size;
	int rowSize;
	int colSize;
	float globalMatrixA[matrixSize][matrixSize];
	float globalMatrixB[matrixSize][matrixSize];
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Request request;
	MPI_Status status;
	int baseCommunicatorRanks[(int) sqrt(size)];
	float localmatrixA[matrixSize/(int)sqrt(size)][matrixSize]; // First scatter buffer that holds all the rows a row of processors will need
	float finalmatrixA[matrixSize/(int)sqrt(size)][matrixSize/(int)sqrt(size)]; // Second scatter buffer which contains the matrix partition for the processor
	float throwAway[matrixSize/(int)sqrt(size)][matrixSize/(int)sqrt(size)]; // Second scatter buffer which contains the matrix partition for the processor
	float localmatrixB[matrixSize/(int)sqrt(size)][matrixSize]; // First scatter buffer that holds all the rows a row of processors will need
	float finalmatrixB[matrixSize/(int)sqrt(size)][matrixSize/(int)sqrt(size)]; // Second scatter buffer which contains the matrix partition for the processor
	float localresult[matrixSize/(int)sqrt(size)][matrixSize/(int)sqrt(size)];
	float rowresult[matrixSize/(int)sqrt(size)][matrixSize];
	float globalResult[matrixSize][matrixSize];
	int sendRankRow;
	int recvRankRow;
	int sendRankCol;
	int recvRankCol;

	double t1, t2;
	
	MPI_Comm row_comm;
	MPI_Comm column_comm;
	int rowColor = rank/(int)sqrt(size);
	int columnColor = rank%(int)sqrt(size);
	/* Custom communicators that will make scattering and gathering matrix information easier. Each processor forms a communicator with the processors
 	 * in the same row and same column */
	MPI_Comm_split(MPI_COMM_WORLD, rowColor, rank, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, columnColor, rank, &column_comm);
	MPI_Comm_size(row_comm, &rowSize);
	MPI_Comm_rank(row_comm, &rowRank);
	MPI_Comm_size(column_comm, &colSize);
	MPI_Comm_rank(column_comm, &colRank);
		
	/* Initialize the matrices with "random" numbers */	
	if(rank == 0){
		int subMatrixDimension = matrixSize / size;
		int subMatrixSize = subMatrixDimension * subMatrixDimension;
		int columnIndex;
		t1 = MPI_Wtime();
		srand(0);

		for(int i = 0; i < matrixSize; i++){
			for(int y = 0; y < matrixSize; y++){
				globalMatrixA[i][y] = ((float)rand() / (float)RAND_MAX) * 2 - 1;
				globalMatrixB[i][y] = ((float)rand() / (float)RAND_MAX) * 2 - 1;
			}
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

	/* Perform initial alignment */

	sendRankRow = mod((rowRank - 1), rowSize);
	recvRankRow = mod((rowRank + 1), rowSize);
	sendRankCol = mod((colRank - 1), rowSize);
	recvRankCol = mod((colRank + 1), rowSize);
	for(int i = 0; i < colRank; i++) {
		MPI_Isend(finalmatrixA, (matrixSize/(int)sqrt(size))*(matrixSize/(int)sqrt(size)), MPI_FLOAT, sendRankRow, 0, row_comm, &request);
		MPI_Recv(finalmatrixA, (matrixSize/(int)sqrt(size))*(matrixSize/(int)sqrt(size)), MPI_FLOAT, recvRankRow, 0, row_comm, &status);
		MPI_Wait(&request, &status);	
	}
	for(int i = 0; i < rowRank; i++) {
		MPI_Isend(finalmatrixB, (matrixSize/(int)sqrt(size))*(matrixSize/(int)sqrt(size)), MPI_FLOAT, sendRankCol, 0, column_comm, &request);
		MPI_Recv(finalmatrixB, (matrixSize/(int)sqrt(size))*(matrixSize/(int)sqrt(size)), MPI_FLOAT, recvRankCol, 0, column_comm, &status);
		MPI_Wait(&request, &status);
	}

	for(int a = 0; a < rowSize; a++) {
		/* Compute the values of the matrix multiplication locally on the processor */
		for(int i = 0; i < matrixSize/(int)sqrt(size); i++) {
			for(int y = 0; y < matrixSize/(int)sqrt(size); y++) {
				if(a == 0) {
					localresult[i][y] = 0;
				}
				for(int z = 0; z < matrixSize/(int)sqrt(size); z++) {
					localresult[i][y] += (finalmatrixA[i][z] * finalmatrixB[z][y]);
				}
			}
		}

		if(a < rowSize - 1) {
			MPI_Isend(finalmatrixA, (matrixSize/(int)sqrt(size))*(matrixSize/(int)sqrt(size)), MPI_FLOAT, sendRankRow, 0, row_comm, &request);
			MPI_Recv(finalmatrixA, (matrixSize/(int)sqrt(size))*(matrixSize/(int)sqrt(size)), MPI_FLOAT, recvRankRow, 0, row_comm, &status);
			MPI_Wait(&request, &status);	
			MPI_Isend(finalmatrixB, (matrixSize/(int)sqrt(size))*(matrixSize/(int)sqrt(size)), MPI_FLOAT, sendRankCol, 0, column_comm, &request);
			MPI_Recv(finalmatrixB, (matrixSize/(int)sqrt(size))*(matrixSize/(int)sqrt(size)), MPI_FLOAT, recvRankCol, 0, column_comm, &status);
			MPI_Wait(&request, &status);	
		} 
	}

	/* With the local result calculated reverse the scatter process and gather up the results */


	for(int i = 0; i < matrixSize/(int)sqrt(size); i++) {	
		MPI_Allgather(localresult[i], matrixSize/(int)sqrt(size), MPI_FLOAT, &rowresult[i], matrixSize/(int)sqrt(size), MPI_FLOAT, row_comm);
	}
	
	/* At this point each row group processor contains all the multiplication results for that row partition. Gather up these results on the root process using the column communicator */
	MPI_Gather(rowresult, (matrixSize*matrixSize)/(int)sqrt(size), MPI_FLOAT, &globalResult, (matrixSize*matrixSize)/(int)sqrt(size), MPI_FLOAT, 0, column_comm);
	
	/* Print out the multiplication result on the world root process just to ensure the results match */
	if(rank == 0) {	
		t2 = MPI_Wtime();
		printf("Matrix multiplication result is\n");
		for(int i = 0; i < matrixSize; i++) {
			for(int y = 0; y < matrixSize; y++) {
				printf(" %.1lf ", globalResult[i][y]);
			}
			printf("\n");
		}
	} 
	MPI_Finalize();
}
