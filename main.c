#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"
#include "help.h"
#include "knn.h"

int main(int argc, char **argv) {
	if (argc < 8) {
		printf("Not enough parameters...\n");
		exit(1);
	}
	int rank, n = atoi(argv[1]), m = atoi(argv[2]), n2 = atoi(argv[3]), k = atoi(argv[4]), noc, numOfProc, i;
	double *xTrain = (double*)malloc(n * m * sizeof(double));
	double *xTest = (double*)malloc(n2 * m * sizeof(double));
	int *yTrain = (int*)malloc(n * sizeof(int));
	int *yTestbuf = (int*)malloc(n2 * sizeof(int));
	int *yTest = (int*)malloc(n2 * sizeof(int));
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		fscanfTrainData(xTrain, yTrain, n, m, argv[5]);
		fscanfTestData(xTest, n2 * n, argv[6]);
		noc = getNumOfClass(yTrain, n);
	}
	MPI_Bcast(&noc, 1, MPI_INT, 0, MPI_COMM_WORLD);
	double _time, a;
	_time = MPI_Wtime();
	if (rank == 0) {
		normalization(xTrain, xTest, n, n2, m);
	}
	MPI_Bcast(xTrain, n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(xTest, n2 * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(yTrain, n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	for (i = rank; i < n2; i += numOfProc) {
		yTestbuf[i] = getClass(xTrain, yTrain, xTest, n, m, k, noc, i);
	}
	MPI_Allreduce(yTestbuf, yTest, n2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	_time = MPI_Wtime() - _time;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		if (argc > 8) {
			int *id = (int*) malloc(n2 * sizeof(int));
			fscanfIdealSpliting(id, n2, argv[8]);
			a = calcAccuracy(yTest, id, n2);
			fprintfFullRes(yTest, n2, a, _time, argv[7]);
			free(id);
			printf("Time for k-NN classification = %lf s.;\nAccuracy of k-NN classification = %lf;\nThe work of the program is completed!\n", _time, a);
		} else {
			fprintfResult(yTest, n2, _time, argv[7]);
			printf("Time for k-NN classification = %lf s.;\nThe work of the program is completed!\n", _time);
		}

	}
	MPI_Finalize();
	free(xTrain);
	free(xTest);
	free(yTrain);
	free(yTest);
	free(yTestbuf);
	return 0;
}
