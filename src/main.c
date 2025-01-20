#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "help.h"
#include "knn.h"


int main(int argc, char **argv) {
	if (argc < 8) {
		puts("Not enough parameters");
		exit(1);
	}
	MPI_Init(&argc, &argv);
	const int n = atoi(argv[3]), m = atoi(argv[4]), n2 = atoi(argv[5]), k = atoi(argv[6]);
	if (n < 1 || m < 1 || n2 < 1 || k < 1 || k > n) {
		puts("Value of parameters is incorrect");
		exit(1);
	}
	double *xTrain = (double*)malloc(n * m * sizeof(double)), *xTest = NULL;
	int pid, *yTrain = (int*)malloc(n * sizeof(int)), *yTest = NULL;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	if (pid == 0) {
		xTest = (double*)malloc(n2 * m * sizeof(double));
		yTest = (int*)malloc(n2 * sizeof(int));
		fscanfTrainData(argv[1], xTrain, yTrain, n, m);
		fscanfTestData(argv[2], xTest, n2 * m);
	}
	MPI_Bcast(yTrain, n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(xTrain, n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	double t = MPI_Wtime();
	MPI_Knn(xTest, yTest, xTrain, yTrain, n2, m, n, k);
	t = MPI_Wtime() - t;
	if (pid == 0) {
		if (argc > 8) {
			int *yIdeal = (int*)malloc(n2 * sizeof(int));
			fscanfIdealSpliting(argv[8], yIdeal, n2);
			const double a = calcAccuracy(yTest, yIdeal, n2);
			fprintfFullRes(argv[7], yTest, n, m, n2, k, a, t);
			free(yIdeal);
			printf("Accuracy of k-NN classification = %lf\nTime for k-NN classification = %lf s.\n", a, t);
		} else {
			fprintfResult(argv[7], yTest, n, m, n2, k, t);
			printf("Time for k-NN classification = %lf s.\n", t);
		}
		free(xTest);
		free(yTest);
	}
	free(xTrain);	
	free(yTrain);	
	MPI_Finalize();
	return 0;
}
